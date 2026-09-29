/*
Copyright The Kubernetes Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"golang.org/x/sys/unix"
)

func TestGetDriver(t *testing.T) {
	t.Run("returns empty driver", func(t *testing.T) {
		pciDevicesPath := t.TempDir()
		pciAddress := "0000:00:01.0"
		devicePath := filepath.Join(pciDevicesPath, pciAddress)
		require.NoError(t, os.MkdirAll(devicePath, 0o755))

		driver, err := getDriver(pciDevicesPath, pciAddress)
		require.NoError(t, err)
		require.Empty(t, driver)
	})

	t.Run("returns valid driver", func(t *testing.T) {
		pciDevicesPath := t.TempDir()
		pciDriversPath := t.TempDir()
		pciAddress := "0000:00:01.0"
		devicePath := filepath.Join(pciDevicesPath, pciAddress)
		require.NoError(t, os.MkdirAll(devicePath, 0o755))

		require.NoError(t, os.Symlink(filepath.Join(pciDriversPath, "nvidia"), filepath.Join(devicePath, "driver")))
		driver, err := getDriver(pciDevicesPath, pciAddress)
		require.NoError(t, err)
		require.Equal(t, "nvidia", driver)
	})
}

func TestTryChangeDriverWithTimeout(t *testing.T) {
	vm := &VfioPciManager{
		nvidiaEnabled: true,
	}

	t.Run("returns success", func(t *testing.T) {
		err := vm.tryChangeDriverWithTimeout(context.Background(), time.Second, func() error {
			return nil
		})
		require.NoError(t, err)
	})

	t.Run("returns error", func(t *testing.T) {
		expected := errors.New("work function failed")
		err := vm.tryChangeDriverWithTimeout(context.Background(), time.Second, func() error {
			return expected
		})
		require.ErrorIs(t, err, expected)
	})

	t.Run("returns on timeout", func(t *testing.T) {
		started := make(chan struct{})
		release := make(chan struct{})
		finished := make(chan struct{})

		err := vm.tryChangeDriverWithTimeout(context.Background(), 10*time.Millisecond, func() error {
			close(started)
			defer close(finished)
			<-release
			return nil
		})
		require.ErrorIs(t, err, context.DeadlineExceeded)
		requireClosed(t, started)

		close(release)
		requireEventuallyClosed(t, finished)
	})

	t.Run("returns on caller cancellation", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		started := make(chan struct{})
		release := make(chan struct{})
		finished := make(chan struct{})

		go func() {
			<-started
			cancel()
		}()

		err := vm.tryChangeDriverWithTimeout(ctx, time.Second, func() error {
			close(started)
			defer close(finished)
			<-release
			return nil
		})
		require.ErrorIs(t, err, context.Canceled)

		close(release)
		requireEventuallyClosed(t, finished)
	})
}

func requireClosed(t *testing.T, ch <-chan struct{}) {
	t.Helper()
	select {
	case <-ch:
	default:
		t.Fatal("expected channel to be closed")
	}
}

func requireEventuallyClosed(t *testing.T, ch <-chan struct{}) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for channel to close")
	}
}

func TestHostRootPersistencedRunning(t *testing.T) {
	t.Run("no host root", func(t *testing.T) {
		vm := &VfioPciManager{nvlib: &deviceLib{}}
		require.False(t, vm.hostRootPersistencedRunning())
	})

	t.Run("socket absent", func(t *testing.T) {
		vm := &VfioPciManager{nvlib: &deviceLib{hostRoot: t.TempDir()}}
		require.False(t, vm.hostRootPersistencedRunning())
	})

	t.Run("socket present", func(t *testing.T) {
		hostRoot := t.TempDir()
		socket := filepath.Join(hostRoot, nvidiaPersistencedSocketPath)
		require.NoError(t, os.MkdirAll(filepath.Dir(socket), 0o755))
		require.NoError(t, os.WriteFile(socket, nil, 0o600))

		vm := &VfioPciManager{nvlib: &deviceLib{hostRoot: hostRoot}}
		require.True(t, vm.hostRootPersistencedRunning())
	})
}

func TestHostRootBinaryPath(t *testing.T) {
	t.Run("returns path relative to the host root", func(t *testing.T) {
		hostRoot := t.TempDir()
		binary := filepath.Join(hostRoot, "usr/local/bin/nvidia-smi")
		require.NoError(t, os.MkdirAll(filepath.Dir(binary), 0o755))
		require.NoError(t, os.WriteFile(binary, nil, 0o755))

		path, err := hostRootBinaryPath(hostRoot, "/usr/local", "nvidia-smi")
		require.NoError(t, err)
		require.Equal(t, "/usr/local/bin/nvidia-smi", path)
	})

	t.Run("binary missing", func(t *testing.T) {
		_, err := hostRootBinaryPath(t.TempDir(), "/usr/local", "nvidia-smi")
		require.Error(t, err)
	})

	t.Run("binary resolves outside the host root", func(t *testing.T) {
		hostRoot, outside := t.TempDir(), t.TempDir()
		target := filepath.Join(outside, "nvidia-smi")
		require.NoError(t, os.WriteFile(target, nil, 0o755))
		link := filepath.Join(hostRoot, "usr/bin/nvidia-smi")
		require.NoError(t, os.MkdirAll(filepath.Dir(link), 0o755))
		require.NoError(t, os.Symlink(target, link))

		_, err := hostRootBinaryPath(hostRoot, "/", "nvidia-smi")
		require.Error(t, err)
	})
}

func TestFindFileUsers(t *testing.T) {
	// newHost builds a fake host root holding the target file and an empty proc.
	newHost := func(t *testing.T) (hostRoot, target string, st unix.Stat_t) {
		t.Helper()
		hostRoot = t.TempDir()
		target = "/dev/nvidia0"
		require.NoError(t, os.MkdirAll(filepath.Join(hostRoot, "dev"), 0o755))
		require.NoError(t, os.WriteFile(filepath.Join(hostRoot, target), nil, 0o600))
		require.NoError(t, os.MkdirAll(filepath.Join(hostRoot, "proc"), 0o755))
		require.NoError(t, unix.Stat(filepath.Join(hostRoot, target), &st))
		return hostRoot, target, st
	}
	addProc := func(t *testing.T, hostRoot, pid, comm string) string {
		t.Helper()
		procDir := filepath.Join(hostRoot, "proc", pid)
		require.NoError(t, os.MkdirAll(filepath.Join(procDir, "fd"), 0o755))
		require.NoError(t, os.WriteFile(filepath.Join(procDir, "comm"), []byte(comm+"\n"), 0o600))
		return procDir
	}

	t.Run("open file descriptor", func(t *testing.T) {
		hostRoot, target, _ := newHost(t)
		procDir := addProc(t, hostRoot, "42", "holder")
		require.NoError(t, os.Symlink(filepath.Join(hostRoot, target), filepath.Join(procDir, "fd", "3")))
		addProc(t, hostRoot, "43", "bystander")

		users, err := findFileUsers(hostRoot, target)
		require.NoError(t, err)
		require.Equal(t, []string{"42 (holder)"}, users)
	})

	t.Run("memory mapping", func(t *testing.T) {
		hostRoot, target, st := newHost(t)
		procDir := addProc(t, hostRoot, "42", "mapper")
		dev := uint64(st.Dev) //nolint:gosec,unconvert
		maps := fmt.Sprintf("7f0000000000-7f0000001000 rw-s 00000000 %x:%x %d %s\n",
			unix.Major(dev), unix.Minor(dev), st.Ino, target)
		require.NoError(t, os.WriteFile(filepath.Join(procDir, "maps"), []byte(maps), 0o600))

		users, err := findFileUsers(hostRoot, target)
		require.NoError(t, err)
		require.Equal(t, []string{"42 (mapper)"}, users)
	})

	t.Run("same device number but a different file", func(t *testing.T) {
		hostRoot, target, _ := newHost(t)
		other := filepath.Join(hostRoot, "other")
		require.NoError(t, os.WriteFile(other, nil, 0o600))
		procDir := addProc(t, hostRoot, "42", "elsewhere")
		require.NoError(t, os.Symlink(other, filepath.Join(procDir, "fd", "3")))

		users, err := findFileUsers(hostRoot, target)
		require.NoError(t, err)
		require.Empty(t, users)
	})

	t.Run("missing path has no users", func(t *testing.T) {
		hostRoot, _, _ := newHost(t)
		users, err := findFileUsers(hostRoot, "/dev/nvidia7")
		require.NoError(t, err)
		require.Empty(t, users)
	})

	t.Run("missing proc", func(t *testing.T) {
		hostRoot, target, _ := newHost(t)
		require.NoError(t, os.RemoveAll(filepath.Join(hostRoot, "proc")))
		_, err := findFileUsers(hostRoot, target)
		require.Error(t, err)
	})
}
