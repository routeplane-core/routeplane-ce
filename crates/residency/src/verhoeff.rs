//! Verhoeff checksum — the algorithm UIDAI uses for the Aadhaar check digit.
//!
//! A 12-digit Aadhaar number's last digit is a Verhoeff check digit over the
//! preceding 11. Validating it lets the classifier reject random 12-digit
//! strings (order ids, transaction refs) that merely match the 4-4-4 shape,
//! sharply cutting false positives (Task #6).

/// Verhoeff multiplication table (D_5 dihedral group).
const D: [[u8; 10]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
];

/// Verhoeff permutation table.
const P: [[u8; 10]; 8] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
];

/// Validate that `digits` (each 0–9) carries a correct Verhoeff check digit as
/// its final element. Returns false for an empty input.
pub fn validate(digits: &[u8]) -> bool {
    if digits.is_empty() {
        return false;
    }
    let mut c: u8 = 0;
    // Process from the rightmost digit (i = 0 is the check digit position).
    for (i, &d) in digits.iter().rev().enumerate() {
        if d > 9 {
            return false;
        }
        c = D[c as usize][P[i % 8][d as usize] as usize];
    }
    c == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_digits(s: &str) -> Vec<u8> {
        s.bytes().map(|b| b - b'0').collect()
    }

    #[test]
    fn known_valid_aadhaar_passes() {
        // A well-known Verhoeff-valid Aadhaar test number.
        assert!(validate(&to_digits("234123412346")));
    }

    #[test]
    fn random_12_digits_fails() {
        assert!(!validate(&to_digits("234123412345")));
        assert!(!validate(&to_digits("111111111111")));
    }

    #[test]
    fn empty_is_invalid() {
        assert!(!validate(&[]));
    }
}
