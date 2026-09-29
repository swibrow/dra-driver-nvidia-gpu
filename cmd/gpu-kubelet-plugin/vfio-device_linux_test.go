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
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestFindFileUsersLiveProc(t *testing.T) {
	path := filepath.Join(t.TempDir(), "held")
	f, err := os.Create(path)
	require.NoError(t, err)
	defer f.Close()

	users, err := findFileUsers("/", path)
	require.NoError(t, err)
	self := fmt.Sprintf("%d ", os.Getpid())
	require.True(t, slices.ContainsFunc(users, func(u string) bool { return strings.HasPrefix(u, self) }),
		"expected %q among users of %s, got %v", self, path, users)
}
