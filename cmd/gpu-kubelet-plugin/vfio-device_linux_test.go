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
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"
)

func TestFindCharDeviceHolders(t *testing.T) {
	self := fmt.Sprintf("%d ", os.Getpid())
	hasSelf := func(holders []string) bool {
		return slices.ContainsFunc(holders, func(h string) bool { return strings.HasPrefix(h, self) })
	}

	f, err := os.Open("/dev/null")
	if err != nil {
		t.Fatalf("open /dev/null: %v", err)
	}
	defer f.Close()

	holders, err := findCharDeviceHolders("/proc", 1, 3)
	if err != nil {
		t.Fatalf("findCharDeviceHolders: %v", err)
	}
	if !hasSelf(holders) {
		t.Errorf("expected %q among holders of 1:3, got %v", self, holders)
	}

	holders, err = findCharDeviceHolders("/proc", nvidiaDeviceMajor, 4095)
	if err != nil {
		t.Fatalf("findCharDeviceHolders: %v", err)
	}
	if hasSelf(holders) {
		t.Errorf("did not expect %q among holders of %d:4095, got %v", self, nvidiaDeviceMajor, holders)
	}

	if _, err := findCharDeviceHolders(t.TempDir()+"/missing", 1, 3); err == nil {
		t.Error("expected an error for a missing proc root")
	}
}
