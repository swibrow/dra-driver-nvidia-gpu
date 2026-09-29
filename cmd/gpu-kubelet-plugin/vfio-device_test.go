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
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
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

func TestPersistencedNvidiaSMI(t *testing.T) {
	writeFile := func(t *testing.T, path string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, nil, 0o755); err != nil {
			t.Fatal(err)
		}
	}

	tests := []struct {
		name          string
		setup         func(t *testing.T, driverRoot, hostRoot string)
		wantHostRoot  bool
		wantNvidiaSMI string
		wantOK        bool
		wantErr       bool
	}{
		{
			name: "socket under driver root",
			setup: func(t *testing.T, driverRoot, _ string) {
				writeFile(t, filepath.Join(driverRoot, nvidiaPersistencedSocketPath))
			},
			wantNvidiaSMI: "nvidia-smi",
			wantOK:        true,
		},
		{
			name: "socket only under host root",
			setup: func(t *testing.T, _, hostRoot string) {
				writeFile(t, filepath.Join(hostRoot, nvidiaPersistencedSocketPath))
				writeFile(t, filepath.Join(hostRoot, "usr/local/bin/nvidia-smi"))
			},
			wantHostRoot:  true,
			wantNvidiaSMI: "/usr/local/bin/nvidia-smi",
			wantOK:        true,
		},
		{
			name: "socket under host root without nvidia-smi",
			setup: func(t *testing.T, _, hostRoot string) {
				writeFile(t, filepath.Join(hostRoot, nvidiaPersistencedSocketPath))
			},
			wantErr: true,
		},
		{
			name:  "not running",
			setup: func(t *testing.T, _, _ string) {},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			driverRoot, hostRoot := t.TempDir(), t.TempDir()
			tc.setup(t, driverRoot, hostRoot)
			vm := &VfioPciManager{
				containerDriverRoot: driverRoot,
				hostDriverRoot:      "/usr/local",
				nvlib:               &deviceLib{devRoot: "/", hostRoot: hostRoot},
			}

			chrootDir, nvidiaSMI, ok, err := vm.persistencedNvidiaSMI()
			if (err != nil) != tc.wantErr {
				t.Fatalf("err = %v, wantErr %v", err, tc.wantErr)
			}
			if ok != tc.wantOK || nvidiaSMI != tc.wantNvidiaSMI {
				t.Errorf("got (%q, %v), want (%q, %v)", nvidiaSMI, ok, tc.wantNvidiaSMI, tc.wantOK)
			}
			wantChroot := ""
			if tc.wantOK {
				wantChroot = "/"
				if tc.wantHostRoot {
					wantChroot = hostRoot
				}
			}
			if chrootDir != wantChroot {
				t.Errorf("chrootDir = %q, want %q", chrootDir, wantChroot)
			}
		})
	}
}
