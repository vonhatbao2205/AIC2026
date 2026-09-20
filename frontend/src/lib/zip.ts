/** Minimal ZIP reader/writer, no dependency.
 *
 * The app ships with React as its only runtime dependency, and both directions
 * needed here are small: read the organiser's query pack (text files, DEFLATE)
 * and write the submission pack (a handful of tiny CSVs).
 *
 * Reading uses the platform `DecompressionStream("deflate-raw")` rather than
 * bundling an inflate implementation. Writing deliberately emits STORED (no
 * compression) entries: a submission zip is a few kilobytes of CSV, so the only
 * thing compression would add is a second code path that can be wrong.
 */

const LOCAL_HEADER_SIG = 0x04034b50;
const CENTRAL_HEADER_SIG = 0x02014b50;
const EOCD_SIG = 0x06054b50;

export interface ZipTextEntry {
  /** Path as stored in the archive, e.g. "query-p1-1-kis.txt". */
  name: string;
  text: string;
}

export class ZipError extends Error {}

function findEocd(view: DataView): number {
  // The EOCD is last, but a trailing comment may follow it; 64 KiB is the most a
  // comment can be, so scanning back that far is exhaustive rather than a guess.
  const start = Math.max(0, view.byteLength - 22 - 0xffff);
  for (let offset = view.byteLength - 22; offset >= start; offset -= 1) {
    if (view.getUint32(offset, true) === EOCD_SIG) return offset;
  }
  throw new ZipError("Invalid ZIP structure (missing End of Central Directory).");
}

async function inflateRaw(data: Uint8Array): Promise<Uint8Array> {
  const Decompressor = (globalThis as { DecompressionStream?: typeof DecompressionStream })
    .DecompressionStream;
  if (!Decompressor) {
    throw new ZipError(
      "This browser does not support DecompressionStream. Use Chrome/Edge 80+, " +
        "Firefox 113+ or Safari 16.4+ to import compressed ZIP files.",
    );
  }
  const stream = new Blob([data as BlobPart]).stream().pipeThrough(new Decompressor("deflate-raw"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Read every file in the archive as UTF-8 text. Directory entries are skipped. */
export async function readZipTextFiles(blob: Blob): Promise<ZipTextEntry[]> {
  const buffer = await blob.arrayBuffer();
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  const eocd = findEocd(view);
  const count = view.getUint16(eocd + 10, true);
  let pointer = view.getUint32(eocd + 16, true);

  const decoder = new TextDecoder("utf-8");
  const entries: ZipTextEntry[] = [];
  for (let index = 0; index < count; index += 1) {
    if (view.getUint32(pointer, true) !== CENTRAL_HEADER_SIG) {
      throw new ZipError(`Invalid central directory at entry ${index + 1}.`);
    }
    const method = view.getUint16(pointer + 10, true);
    const compressedSize = view.getUint32(pointer + 20, true);
    const nameLength = view.getUint16(pointer + 28, true);
    const extraLength = view.getUint16(pointer + 30, true);
    const commentLength = view.getUint16(pointer + 32, true);
    const localOffset = view.getUint32(pointer + 42, true);
    const name = decoder.decode(bytes.subarray(pointer + 46, pointer + 46 + nameLength));
    pointer += 46 + nameLength + extraLength + commentLength;

    if (name.endsWith("/")) continue; // directory entry
    if (view.getUint32(localOffset, true) !== LOCAL_HEADER_SIG) {
      throw new ZipError(`Invalid local header for ${name}.`);
    }
    // The local header repeats the name/extra lengths and they may differ from
    // the central copy, so the data offset must be computed from the local one.
    const localNameLength = view.getUint16(localOffset + 26, true);
    const localExtraLength = view.getUint16(localOffset + 28, true);
    const dataStart = localOffset + 30 + localNameLength + localExtraLength;
    const raw = bytes.subarray(dataStart, dataStart + compressedSize);

    let content: Uint8Array;
    if (method === 0) content = raw;
    else if (method === 8) content = await inflateRaw(raw);
    else throw new ZipError(`${name}: compression method ${method} is not supported.`);
    entries.push({ name, text: decoder.decode(content) });
  }
  return entries;
}

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

export function crc32(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < data.length; i += 1) crc = CRC_TABLE[(crc ^ data[i]) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

export interface ZipOutputFile {
  name: string;
  text: string;
}

/** Build an uncompressed (STORED) zip as raw bytes. */
export function buildZipBytes(files: ZipOutputFile[]): Uint8Array {
  const encoder = new TextEncoder();
  const locals: Uint8Array[] = [];
  const centrals: Uint8Array[] = [];
  let offset = 0;

  for (const file of files) {
    const nameBytes = encoder.encode(file.name);
    const data = encoder.encode(file.text);
    const crc = crc32(data);

    const local = new Uint8Array(30 + nameBytes.length + data.length);
    const localView = new DataView(local.buffer);
    localView.setUint32(0, LOCAL_HEADER_SIG, true);
    localView.setUint16(4, 20, true); // version needed
    localView.setUint16(6, 0x0800, true); // UTF-8 filename flag
    localView.setUint16(8, 0, true); // STORED
    localView.setUint16(10, 0, true); // mod time — fixed, so exports are reproducible
    localView.setUint16(12, 0x0021, true); // mod date = 1980-01-01
    localView.setUint32(14, crc, true);
    localView.setUint32(18, data.length, true);
    localView.setUint32(22, data.length, true);
    localView.setUint16(26, nameBytes.length, true);
    localView.setUint16(28, 0, true);
    local.set(nameBytes, 30);
    local.set(data, 30 + nameBytes.length);
    locals.push(local);

    const central = new Uint8Array(46 + nameBytes.length);
    const centralView = new DataView(central.buffer);
    centralView.setUint32(0, CENTRAL_HEADER_SIG, true);
    centralView.setUint16(4, 20, true);
    centralView.setUint16(6, 20, true);
    centralView.setUint16(8, 0x0800, true);
    centralView.setUint16(10, 0, true);
    centralView.setUint16(12, 0, true);
    centralView.setUint16(14, 0x0021, true);
    centralView.setUint32(16, crc, true);
    centralView.setUint32(20, data.length, true);
    centralView.setUint32(24, data.length, true);
    centralView.setUint16(28, nameBytes.length, true);
    centralView.setUint32(42, offset, true);
    central.set(nameBytes, 46);
    centrals.push(central);

    offset += local.length;
  }

  const centralSize = centrals.reduce((sum, part) => sum + part.length, 0);
  const eocd = new Uint8Array(22);
  const eocdView = new DataView(eocd.buffer);
  eocdView.setUint32(0, EOCD_SIG, true);
  eocdView.setUint16(8, files.length, true);
  eocdView.setUint16(10, files.length, true);
  eocdView.setUint32(12, centralSize, true);
  eocdView.setUint32(16, offset, true);

  const parts = [...locals, ...centrals, eocd];
  const out = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
  let at = 0;
  for (const part of parts) {
    out.set(part, at);
    at += part.length;
  }
  return out;
}

export function buildZip(files: ZipOutputFile[]): Blob {
  return new Blob([buildZipBytes(files) as BlobPart], { type: "application/zip" });
}
