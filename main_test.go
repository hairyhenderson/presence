package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestShiftPresence(t *testing.T) {
	presence := uint8(0)

	record(true, &presence)
	assert.Equal(t, uint8(0b0000_0001), presence)

	record(false, &presence)
	assert.Equal(t, uint8(0b0000_0010), presence)

	presence = 0b1111_1111
	record(true, &presence)
	assert.Equal(t, uint8(0b1111_1111), presence)

	record(false, &presence)
	assert.Equal(t, uint8(0b1111_1110), presence)

	// 7 more false records should clear it
	record(false, &presence)
	record(false, &presence)
	record(false, &presence)
	record(false, &presence)
	record(false, &presence)
	record(false, &presence)
	record(false, &presence)
	assert.Equal(t, uint8(0b0000_000), presence)
}
