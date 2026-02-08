package simulator

import (
	"time"

	loom "github.com/LegendaryPenguin/loom/go-control-plane"
)

// Arrival is one request plus the one fact the control plane is not allowed to
// know.
//
// ActualOutputTokens is where generation really stops. The whole admission problem
// is that this number exists and is invisible: the client declares
// MaxOutputTokens, the control plane has to reserve memory against that ceiling or
// against a prediction of this, and the gap between the two is what every
// reservation-mode and cache-policy result in bench/ turns on. It lives here, in the
// simulator's input, and is never read by anything in go-control-plane.
type Arrival struct {
	Req loom.Request

	// ActualOutputTokens must be in [1, MaxOutputTokens]. A generator that exceeds
	// the ceiling is modelling something that cannot happen.
	ActualOutputTokens int

	// Offset is the arrival instant relative to the start of the run. Relative rather
	// than absolute so a workload can be generated once and replayed against any
	// simulation start time.
	Offset time.Duration

	// DeadlineAfter, when non-zero, is the client's budget measured from arrival. Carried
	// relative for the same reason Offset is, and converted to the absolute
	// Request.Deadline the control plane wants at the moment the request arrives.
	DeadlineAfter time.Duration
}
