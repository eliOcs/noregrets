//! Cards, ranks and suits. Each is a validated newtype over a `u8`, so a
//! value of these types is always a real card, rank or suit: the only ways
//! to make one are the checked constructors and the `ALL` tables.
//!
//! A card is encoded rank*4 + suit, which keeps it one byte and makes
//! `index()` usable directly for 52-entry tables.
//! Ranks: 0 = Two .. 12 = Ace. Suits: 0=c, 1=d, 2=h, 3=s.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

pub const RANK_CHARS: [char; 13] = [
    '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
];
pub const SUIT_CHARS: [char; 4] = ['c', 'd', 'h', 's'];

/// A card rank, Two through Ace, ordered by strength.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Rank(u8);

impl Rank {
    pub const TWO: Rank = Rank(0);
    pub const ACE: Rank = Rank(12);

    /// Every rank, Two first.
    pub const ALL: [Rank; 13] = {
        let mut all = [Rank(0); 13];
        let mut i = 0;
        while i < 13 {
            all[i] = Rank(i as u8);
            i += 1;
        }
        all
    };

    /// The rank with this index (0 = Two .. 12 = Ace), if there is one.
    pub const fn new(index: u8) -> Option<Rank> {
        if index < 13 {
            Some(Rank(index))
        } else {
            None
        }
    }

    /// 0 = Two .. 12 = Ace.
    #[inline(always)]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Single-bit mask of this rank, for rank-set bitboards.
    #[inline(always)]
    pub const fn bit(self) -> u16 {
        1 << self.0
    }

    pub const fn to_char(self) -> char {
        RANK_CHARS[self.0 as usize]
    }

    pub fn from_char(c: char) -> Option<Rank> {
        let c = c.to_ascii_uppercase();
        RANK_CHARS.iter().position(|&r| r == c).map(|i| Rank(i as u8))
    }
}

/// A card suit. Suits have no order of strength; the order is only the
/// encoding (clubs, diamonds, hearts, spades).
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Suit(u8);

impl Suit {
    pub const ALL: [Suit; 4] = [Suit(0), Suit(1), Suit(2), Suit(3)];

    /// The suit with this index (0=c, 1=d, 2=h, 3=s), if there is one.
    pub const fn new(index: u8) -> Option<Suit> {
        if index < 4 {
            Some(Suit(index))
        } else {
            None
        }
    }

    /// 0=c, 1=d, 2=h, 3=s.
    #[inline(always)]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn to_char(self) -> char {
        SUIT_CHARS[self.0 as usize]
    }

    pub fn from_char(c: char) -> Option<Suit> {
        let c = c.to_ascii_lowercase();
        SUIT_CHARS.iter().position(|&s| s == c).map(|i| Suit(i as u8))
    }
}

/// One of the 52 cards. Serialized as its index byte; deserializing
/// rejects bytes that are not a card.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub struct Card(u8);

impl Card {
    /// Every card in index order (2c 2d 2h 2s 3c .. As).
    pub const ALL: [Card; 52] = {
        let mut all = [Card(0); 52];
        let mut i = 0;
        while i < 52 {
            all[i] = Card(i as u8);
            i += 1;
        }
        all
    };

    /// Initial value for fixed-size card buffers (decks, boards being
    /// dealt) whose meaningful length is tracked alongside them.
    pub const FILL: Card = Card(0);

    #[inline(always)]
    pub const fn new(rank: Rank, suit: Suit) -> Card {
        Card((rank.0 << 2) | suit.0)
    }

    /// The card with this index (rank*4 + suit), if there is one.
    #[inline(always)]
    pub const fn from_index(index: usize) -> Option<Card> {
        if index < 52 {
            Some(Card(index as u8))
        } else {
            None
        }
    }

    /// rank*4 + suit, in 0..52: the card's slot in 52-entry tables.
    #[inline(always)]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    #[inline(always)]
    pub const fn rank(self) -> Rank {
        Rank(self.0 >> 2)
    }

    #[inline(always)]
    pub const fn suit(self) -> Suit {
        Suit(self.0 & 3)
    }

    /// The card of the same rank in another suit.
    #[inline(always)]
    pub const fn with_suit(self, suit: Suit) -> Card {
        Card((self.0 & !3) | suit.0)
    }
}

impl From<Card> for u8 {
    fn from(c: Card) -> u8 {
        c.0
    }
}

impl TryFrom<u8> for Card {
    type Error = String;

    fn try_from(b: u8) -> Result<Card, String> {
        Card::from_index(b as usize).ok_or_else(|| format!("{b} is not a card index (0..52)"))
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.rank().to_char(), self.suit().to_char())
    }
}

impl FromStr for Card {
    type Err = String;

    /// Parse a card like "As", "Td", "2c".
    fn from_str(s: &str) -> Result<Card, String> {
        parse_card(s).ok_or_else(|| format!("bad card '{s}'"))
    }
}

pub fn card_str(c: Card) -> String {
    c.to_string()
}

pub fn cards_str(cards: &[Card]) -> String {
    cards
        .iter()
        .map(|&c| card_str(c))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Parse a card like "As", "Td", "2c". Returns None on bad input.
pub fn parse_card(s: &str) -> Option<Card> {
    let mut it = s.chars();
    let r = Rank::from_char(it.next()?)?;
    let u = Suit::from_char(it.next()?)?;
    if it.next().is_some() {
        return None;
    }
    Some(Card::new(r, u))
}

/// Parse space-separated cards, e.g. "As Kd 7c".
#[cfg(test)]
pub fn parse_cards(s: &str) -> Option<Vec<Card>> {
    s.split_whitespace().map(parse_card).collect()
}

/// A fresh 52-card deck in order; shuffle with your own RNG.
pub fn fresh_deck() -> [Card; 52] {
    Card::ALL
}

/// Shorthand for building cards from rank and suit indices in tests.
#[cfg(test)]
pub fn make_card(rank: u8, suit: u8) -> Card {
    Card::new(Rank::new(rank).unwrap(), Suit::new(suit).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        for c in Card::ALL {
            assert_eq!(parse_card(&card_str(c)), Some(c));
        }
    }

    #[test]
    fn parse_examples() {
        assert_eq!(parse_card("As"), Some(make_card(12, 3)));
        assert_eq!(parse_card("2c"), Some(make_card(0, 0)));
        assert_eq!(parse_card("Td"), Some(make_card(8, 1)));
        assert_eq!(parse_card("xx"), None);
    }

    #[test]
    fn only_real_cards_exist() {
        assert_eq!(Card::from_index(51).map(card_str).as_deref(), Some("As"));
        assert_eq!(Card::from_index(52), None);
        assert_eq!(Rank::new(13), None);
        assert_eq!(Suit::new(4), None);
        assert!(Card::try_from(52u8).is_err());
        let c: Card = bincode::deserialize(&[51u8]).unwrap();
        assert_eq!(c, make_card(12, 3));
        assert!(bincode::deserialize::<Card>(&[200u8]).is_err());
        assert_eq!(bincode::serialize(&c).unwrap(), vec![51u8]);
    }
}
