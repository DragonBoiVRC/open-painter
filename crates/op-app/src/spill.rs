//! The paint of a texture set that is not on the GPU: kept in memory, or in a file when memory runs short.

use op_project::{BlobKind, TextureBlob};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

/// Per UV tile, the blobs of its textures.
pub type Tiles = Vec<Vec<TextureBlob>>;

const KINDS: [BlobKind; 7] = [BlobKind::Base, BlobKind::Metal, BlobKind::Rough, BlobKind::Height, BlobKind::Emissive, BlobKind::Opacity, BlobKind::Mask];

pub enum Saved {
    Ram(Tiles),
    /// A file we wrote; it is deleted with this value.
    Disk(PathBuf),
}

impl Saved {
    pub fn ram_bytes(&self) -> u64 {
        match self {
            Saved::Ram(t) => t.iter().flatten().map(|b| b.bytes.len() as u64).sum(),
            Saved::Disk(_) => 0,
        }
    }

    /// A copy of the blobs (read back from the file if they were spilled).
    pub fn blobs(&self) -> io::Result<Tiles> {
        match self {
            Saved::Ram(t) => Ok(t.clone()),
            Saved::Disk(p) => read(&mut io::BufReader::new(std::fs::File::open(p)?)),
        }
    }

    /// Move the blobs out of memory into `dir/name`. A failed write leaves them in memory.
    pub fn spill(&mut self, dir: &Path, name: &str) -> io::Result<()> {
        let Saved::Ram(tiles) = self else { return Ok(()) };
        std::fs::create_dir_all(dir)?;
        let path = dir.join(name);
        let mut w = io::BufWriter::new(std::fs::File::create(&path)?);
        let written = write(&mut w, tiles).and_then(|()| w.flush());
        if let Err(e) = written {
            let _ = std::fs::remove_file(&path);
            return Err(e);
        }
        *self = Saved::Disk(path);
        Ok(())
    }
}

impl Drop for Saved {
    fn drop(&mut self) {
        if let Saved::Disk(p) = self {
            let _ = std::fs::remove_file(p);
        }
    }
}

fn write(w: &mut impl Write, tiles: &Tiles) -> io::Result<()> {
    w.write_all(&(tiles.len() as u32).to_le_bytes())?;
    for tile in tiles {
        w.write_all(&(tile.len() as u32).to_le_bytes())?;
        for b in tile {
            let kind = KINDS.iter().position(|k| *k == b.kind).unwrap_or(0) as u8;
            w.write_all(&b.node.to_le_bytes())?;
            w.write_all(&[kind])?;
            w.write_all(&(b.bytes.len() as u64).to_le_bytes())?;
            w.write_all(&b.bytes)?;
        }
    }
    Ok(())
}

fn read(r: &mut impl Read) -> io::Result<Tiles> {
    fn num<const N: usize>(r: &mut impl Read) -> io::Result<[u8; N]> {
        let mut b = [0u8; N];
        r.read_exact(&mut b)?;
        Ok(b)
    }
    let bad = || io::Error::new(io::ErrorKind::InvalidData, "spill file is damaged");
    let mut tiles = Vec::new();
    for _ in 0..u32::from_le_bytes(num(r)?) {
        let mut tile = Vec::new();
        for _ in 0..u32::from_le_bytes(num(r)?) {
            let node = u64::from_le_bytes(num(r)?);
            let kind = *KINDS.get(num::<1>(r)?[0] as usize).ok_or_else(bad)?;
            let mut bytes = vec![0u8; u64::from_le_bytes(num(r)?) as usize];
            r.read_exact(&mut bytes)?;
            tile.push(TextureBlob { node, kind, bytes });
        }
        tiles.push(tile);
    }
    Ok(tiles)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spilled_paint_comes_back_and_the_file_goes_with_it() {
        let dir = std::env::temp_dir().join("op-spill-test");
        let tiles = vec![vec![TextureBlob { node: 7, kind: BlobKind::Rough, bytes: vec![1, 2, 3] }], vec![]];
        let mut s = Saved::Ram(tiles.clone());
        s.spill(&dir, "a.bin").unwrap();
        assert_eq!(s.ram_bytes(), 0);
        assert_eq!(s.blobs().unwrap(), tiles);
        drop(s);
        assert!(!dir.join("a.bin").exists());
    }
}
