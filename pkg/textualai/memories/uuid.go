package memories

import (
	"crypto/rand"
	"fmt"
)

type UUID string

// NoUUID is an undefined UUID
const NoUUID = UUID("NoUUID")

func (i UUID) String() string {
	return string(i)
}

// V4UUID generates a random UUID, according to RFC 4122
func V4UUID() UUID {
	b := make([]byte, 16)
	_, err := rand.Read(b) // we use crypto/rand to generate a random UUID
	if err != nil {
		return ""
	}
	b[6] = (b[6] & 0x0f) | 0x40 // set version to 4
	b[8] = (b[8] & 0x3f) | 0x80 // set variant to 10 == 2
	return UUID(fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:]))
}
