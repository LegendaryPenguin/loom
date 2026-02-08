package bench

import "fmt"

func f1(v float64) string { return fmt.Sprintf("%.1f", v) }
func f2(v float64) string { return fmt.Sprintf("%.2f", v) }
func f3(v float64) string { return fmt.Sprintf("%.3f", v) }

func itoa(n int) string      { return fmt.Sprintf("%d", n) }
func itoa64(n uint64) string { return fmt.Sprintf("%d", n) }
