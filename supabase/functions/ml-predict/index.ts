import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { inflate, inflateRaw } from "https://esm.sh/pako@2.1.0?target=deno";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

interface PredictRequest {
  bbox: {
    minLat: number;
    maxLat: number;
    minLng: number;
    maxLng: number;
  };
  startDate?: string;
  endDate?: string;
  modelName?: string;
  modelVersion?: string;
  forceNdvi?: boolean;
  threshold?: number;
}

interface SentinelToken {
  access_token: string;
  expires_in: number;
}

type SentinelProvider = "cdse" | "sentinelhub";

type SentinelAuthConfig = {
  provider: SentinelProvider;
  tokenUrl: string;
  processUrl: string;
};

function getSentinelAuthConfig(provider: SentinelProvider): SentinelAuthConfig {
  if (provider === "sentinelhub") {
    return {
      provider,
      tokenUrl: "https://services.sentinel-hub.com/oauth/token",
      // Use US-West-2 endpoint which supports Landsat 8-9 L2 data
      processUrl: "https://services-uswest2.sentinel-hub.com/api/v1/process",
    };
  }

  return {
    provider,
    tokenUrl:
      "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
    processUrl: "https://sh.dataspace.copernicus.eu/api/v1/process",
  };
}

// Get OAuth token (CDSE by default; fallback to Sentinel Hub)
async function getSentinelToken(): Promise<{ accessToken: string; processUrl: string; provider: SentinelProvider }> {
  const clientId = Deno.env.get("SENTINEL_CLIENT_ID");
  const clientSecret = Deno.env.get("SENTINEL_CLIENT_SECRET");

  if (!clientId || !clientSecret) {
    throw new Error("Sentinel credentials not configured");
  }

  const tryProvider = async (provider: SentinelProvider) => {
    const cfg = getSentinelAuthConfig(provider);

    const tokenResponse = await fetch(cfg.tokenUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({
        grant_type: "client_credentials",
        client_id: clientId,
        client_secret: clientSecret,
      }),
    });

    if (!tokenResponse.ok) {
      const errorText = await tokenResponse.text();
      console.error(`Token error (${provider}):`, errorText);
      return { ok: false as const, status: tokenResponse.status, errorText, provider };
    }

    const tokenData: SentinelToken = await tokenResponse.json();
    return {
      ok: true as const,
      accessToken: tokenData.access_token,
      processUrl: cfg.processUrl,
      provider,
    };
  };

  // 1) Try Copernicus Data Space (CDSE)
  const cdse = await tryProvider("cdse");
  if (cdse.ok) return { accessToken: cdse.accessToken, processUrl: cdse.processUrl, provider: cdse.provider };

  // 2) Fallback: try Sentinel Hub (common confusion where users provide Sentinel Hub OAuth client)
  const sh = await tryProvider("sentinelhub");
  if (sh.ok) return { accessToken: sh.accessToken, processUrl: sh.processUrl, provider: sh.provider };

  // If both fail, keep original (most informative) error
  throw new Error(`Failed to get Sentinel token: ${cdse.status}`);
}

// ============= LANDSAT 8 BAND MAPPING =============
// Landsat 8 Collection 2 Level-2 bands:
// SR_B2 = Blue, SR_B3 = Green, SR_B4 = Red
// SR_B5 = NIR, SR_B6 = SWIR1, SR_B7 = SWIR2

const LANDSAT8_BANDS = {
  Blue: 'B02',
  Green: 'B03',
  Red: 'B04',
  NIR: 'B05',
  SWIR1: 'B06',
  SWIR2: 'B07',
};

// GEE preprocessing offset
// GEE applies: DN * 0.0000275 - 0.2 to raw Landsat SR values
// Sentinel Hub returns reflectance (0-1), we need to apply the -0.2 offset
// to match the GEE-scaled training data range: [-0.2, 0.6]
const GEE_OFFSET = -0.2;

function applyGeeOffset(reflectance: number): number {
  // Apply the -0.2 offset to match GEE-scaled surface reflectance
  return reflectance + GEE_OFFSET;
}

// Fetch a single Landsat 8 band as FLOAT32 array
// Returns RAW reflectance (0-1 range) - GEE offset applied later
async function fetchLandsat8BandData(
  accessToken: string,
  processUrl: string,
  bbox: number[],
  band: string,
  startDate: string,
  endDate: string,
  width: number,
  height: number
): Promise<Float32Array> {
  const evalscript = `
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["${band}"],
      units: "REFLECTANCE"
    }],
    output: {
      bands: 1,
      sampleType: "INT16"
    }
  };
}

function evaluatePixel(sample) {
  // Scale reflectance to INT16: multiply by 10000
  let val = sample.${band};
  val = Math.max(-3.2, Math.min(3.2, val));
  return [Math.round(val * 10000)];
}
`;

  const requestBody = {
    input: {
      bounds: {
        bbox: bbox,
        properties: {
          crs: "http://www.opengis.net/def/crs/EPSG/0/4326",
        },
      },
      data: [
        {
          type: "landsat-ot-l2",
          dataFilter: {
            timeRange: {
              from: startDate,
              to: endDate,
            },
            mosaickingOrder: "leastCC",
          },
        },
      ],
    },
    output: {
      width: width,
      height: height,
      responses: [
        {
          identifier: "default",
          format: { 
            type: "image/tiff",
            compression: "NONE"
          },
        },
      ],
    },
    evalscript: evalscript,
  };

  const response = await fetch(processUrl, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
      Accept: "image/tiff",
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error(`Band ${band} fetch error:`, response.status, errorText);
    throw new Error(`Failed to fetch Landsat 8 band ${band}: ${response.status}`);
  }

  const arrayBuffer = await response.arrayBuffer();
  console.log(`Band ${band}: received ${arrayBuffer.byteLength} bytes`);
  
  // Parse TIFF - returns reflectance values (0-1 range)
  const reflectanceData = await parseTiffInt16ToFloat32(arrayBuffer, width, height, band, 10000);
  
  return reflectanceData;
}

// Parse TIFF with INT16 data, handling LZW and Deflate compression
async function parseTiffInt16ToFloat32(
  buffer: ArrayBuffer,
  width: number,
  height: number,
  band: string,
  divisor: number = 10000
): Promise<Float32Array> {
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  const totalPixels = width * height;
  
  if (buffer.byteLength < 8) {
    throw new Error('Buffer too small for TIFF');
  }
  
  const byte0 = view.getUint8(0);
  const byte1 = view.getUint8(1);
  const littleEndian = (byte0 === 0x49 && byte1 === 0x49);
  
  if (byte0 !== 0x49 && byte0 !== 0x4D) {
    throw new Error('Invalid TIFF header');
  }
  
  const ifdOffset = view.getUint32(4, littleEndian);
  if (ifdOffset + 2 > buffer.byteLength) {
    throw new Error('Invalid IFD offset');
  }
  
  const numEntries = view.getUint16(ifdOffset, littleEndian);
  
  let compression = 1;
  let stripOffsets: number[] = [];
  let stripByteCounts: number[] = [];
  let rowsPerStrip = height;
  
  for (let i = 0; i < numEntries; i++) {
    const entryOffset = ifdOffset + 2 + i * 12;
    if (entryOffset + 12 > buffer.byteLength) break;
    
    const tag = view.getUint16(entryOffset, littleEndian);
    const type = view.getUint16(entryOffset + 2, littleEndian);
    const count = view.getUint32(entryOffset + 4, littleEndian);
    const valueField = entryOffset + 8;
    
    const getValue = () => type === 3 ? view.getUint16(valueField, littleEndian) : view.getUint32(valueField, littleEndian);
    const getPointer = () => view.getUint32(valueField, littleEndian);

    const readOffsets = (ptr: number, n: number): number[] => {
      const out: number[] = [];
      const itemSize = type === 3 ? 2 : 4;
      for (let j = 0; j < n && ptr + j * itemSize + itemSize <= buffer.byteLength; j++) {
        out.push(type === 3
          ? view.getUint16(ptr + j * itemSize, littleEndian)
          : view.getUint32(ptr + j * itemSize, littleEndian)
        );
      }
      return out;
    };

    switch (tag) {
      case 259: compression = getValue(); break;
      case 278: rowsPerStrip = getValue(); break;
      case 273: // StripOffsets
        if (count === 1) {
          stripOffsets = [getValue()];
        } else {
          const ptr = getPointer();
          stripOffsets = readOffsets(ptr, count);
        }
        break;
      case 279: // StripByteCounts
        if (count === 1) {
          stripByteCounts = [getValue()];
        } else {
          const ptr = getPointer();
          stripByteCounts = readOffsets(ptr, count);
        }
        break;
    }
  }
  
  console.log(`${band} TIFF: compression=${compression}, strips=${stripOffsets.length}`);
  
  const result = new Float32Array(totalPixels);
  
  if (compression === 5) {
    // LZW compression
    let pixelIndex = 0;
    for (let s = 0; s < stripOffsets.length && pixelIndex < totalPixels; s++) {
      const stripData = decompressLZW(bytes, stripOffsets[s], stripByteCounts[s]);
      const stripView = new DataView(stripData.buffer);
      const pixelCount = Math.floor(stripData.length / 2);
      
      for (let i = 0; i < pixelCount && pixelIndex < totalPixels; i++) {
        const int16Val = stripView.getInt16(i * 2, littleEndian);
           result[pixelIndex++] = int16Val / divisor;
      }
    }
  } else if (compression === 1) {
    // Uncompressed
    let pixelIndex = 0;
    for (let s = 0; s < stripOffsets.length && pixelIndex < totalPixels; s++) {
      const offset = stripOffsets[s];
      const byteCount = stripByteCounts[s] || (totalPixels * 2);
      const pixelCount = Math.floor(byteCount / 2);
      
      for (let i = 0; i < pixelCount && pixelIndex < totalPixels; i++) {
        const pos = offset + i * 2;
        if (pos + 2 <= buffer.byteLength) {
          const int16Val = view.getInt16(pos, littleEndian);
          result[pixelIndex++] = int16Val / divisor;
        }
      }
    }
  } else if (compression === 8 || compression === 32946) {
    // Deflate/ZLIB compression
    console.log(`Deflate compression detected, attempting decompression`);
    let pixelIndex = 0;
    
    for (let s = 0; s < stripOffsets.length && pixelIndex < totalPixels; s++) {
      try {
        const compressedData = bytes.slice(stripOffsets[s], stripOffsets[s] + stripByteCounts[s]);
        const decompressed = await decompressDeflate(compressedData);
        const stripView = new DataView(decompressed.buffer);
        const pixelCount = Math.floor(decompressed.length / 2);
        
        for (let i = 0; i < pixelCount && pixelIndex < totalPixels; i++) {
           const int16Val = stripView.getInt16(i * 2, littleEndian);
           result[pixelIndex++] = int16Val / divisor;
        }
      } catch (e) {
        console.warn(`Strip ${s} decompression failed:`, e);
        const expectedPixels = Math.ceil(rowsPerStrip * width);
        for (let i = 0; i < expectedPixels && pixelIndex < totalPixels; i++) {
          result[pixelIndex++] = 0;
        }
      }
    }
    
    if (pixelIndex < totalPixels * 0.5) {
      console.warn(`Only ${pixelIndex}/${totalPixels} pixels extracted, using fallback`);
      return extractFallback(buffer, width, height, littleEndian, divisor);
    }
  } else {
    console.warn(`Unknown compression ${compression}, using fallback`);
    return extractFallback(buffer, width, height, littleEndian, divisor);
  }
  
  return result;
}

// LZW decompression for TIFF
function decompressLZW(input: Uint8Array, offset: number, length: number): Uint8Array {
  const output: number[] = [];
  const dictionary: number[][] = [];
  
  for (let i = 0; i < 256; i++) {
    dictionary[i] = [i];
  }
  
  const CLEAR_CODE = 256;
  const EOI_CODE = 257;
  let nextCode = 258;
  let codeSize = 9;
  
  let bitBuffer = 0;
  let bitsInBuffer = 0;
  let bytePos = offset;
  const endPos = offset + length;
  
  const readCode = (): number => {
    while (bitsInBuffer < codeSize && bytePos < endPos) {
      bitBuffer = (bitBuffer << 8) | input[bytePos++];
      bitsInBuffer += 8;
    }
    if (bitsInBuffer < codeSize) return EOI_CODE;
    
    bitsInBuffer -= codeSize;
    return (bitBuffer >> bitsInBuffer) & ((1 << codeSize) - 1);
  };
  
  let oldCode = -1;
  
  while (bytePos < endPos || bitsInBuffer >= codeSize) {
    const code = readCode();
    
    if (code === EOI_CODE) break;
    
    if (code === CLEAR_CODE) {
      dictionary.length = 258;
      for (let i = 0; i < 256; i++) dictionary[i] = [i];
      nextCode = 258;
      codeSize = 9;
      oldCode = -1;
      continue;
    }
    
    let entry: number[];
    if (code < dictionary.length) {
      entry = dictionary[code];
    } else if (code === nextCode && oldCode >= 0) {
      entry = [...dictionary[oldCode], dictionary[oldCode][0]];
    } else {
      console.warn(`Invalid LZW code ${code}`);
      break;
    }
    
    output.push(...entry);
    
    if (oldCode >= 0 && nextCode < 4096) {
      dictionary[nextCode++] = [...dictionary[oldCode], entry[0]];
      if (nextCode >= (1 << codeSize) && codeSize < 12) {
        codeSize++;
      }
    }
    
    oldCode = code;
  }
  
  return new Uint8Array(output);
}

// Decompress Deflate/ZLIB compressed data
async function decompressDeflate(data: Uint8Array): Promise<Uint8Array> {
  try {
    return inflateRaw(data);
  } catch {
    return inflate(data);
  }
}

// Fallback extraction when compression isn't handled
function extractFallback(
  buffer: ArrayBuffer,
  width: number,
  height: number,
  littleEndian: boolean,
  divisor: number
): Float32Array {
  const view = new DataView(buffer);
  const totalPixels = width * height;
  const result = new Float32Array(totalPixels);
  
  const expectedBytes = totalPixels * 2;
  const possibleStarts = [8, 256, 512, buffer.byteLength - expectedBytes];
  
  for (const start of possibleStarts) {
    if (start >= 0 && start + expectedBytes <= buffer.byteLength) {
      let validSamples = 0;
      for (let i = 0; i < Math.min(100, totalPixels); i++) {
        const val = view.getInt16(start + i * 2, littleEndian) / divisor;
        // For divisor=1 (DN) values are typically 0..10000
        // For divisor=10000 (reflectance) values are typically -1..2
        if ((divisor === 1 && val >= -500 && val <= 12000) || (divisor !== 1 && val >= -1 && val <= 2)) validSamples++;
      }
      
      if (validSamples > 80) {
        for (let i = 0; i < totalPixels; i++) {
          result[i] = view.getInt16(start + i * 2, littleEndian) / divisor;
        }
        console.log(`Fallback: found valid data at offset ${start}`);
        return result;
      }
    }
  }
  
  console.warn('Could not extract raster data, returning zeros');
  return result;
}

// Convert Uint8Array to base64 (chunked to avoid call-stack overflow)
function uint8ToBase64(bytes: Uint8Array): string {
  let binary = '';
  const chunkSize = 8192;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}

// Convert Float32Array to base64
function float32ArrayToBase64(array: Float32Array): string {
  return uint8ToBase64(new Uint8Array(array.buffer));
}

// ============= SPECTRAL INDEX CALCULATIONS =============
// These match the GEE training preprocessing pipeline exactly
// Indices are calculated AFTER GEE scaling is applied

// NDVI = (NIR - Red) / (NIR + Red)
function calculateNDVI(nir: Float32Array, red: Float32Array): Float32Array {
  const result = new Float32Array(nir.length);
  for (let i = 0; i < nir.length; i++) {
    const sum = nir[i] + red[i];
    result[i] = sum !== 0 ? (nir[i] - red[i]) / sum : 0;
  }
  return result;
}

// NDWI = (Green - NIR) / (Green + NIR)
function calculateNDWI(green: Float32Array, nir: Float32Array): Float32Array {
  const result = new Float32Array(green.length);
  for (let i = 0; i < green.length; i++) {
    const sum = green[i] + nir[i];
    result[i] = sum !== 0 ? (green[i] - nir[i]) / sum : 0;
  }
  return result;
}

// NBR = (NIR - SWIR2) / (NIR + SWIR2)
function calculateNBR(nir: Float32Array, swir2: Float32Array): Float32Array {
  const result = new Float32Array(nir.length);
  for (let i = 0; i < nir.length; i++) {
    const sum = nir[i] + swir2[i];
    result[i] = sum !== 0 ? (nir[i] - swir2[i]) / sum : 0;
  }
  return result;
}

// Replace NaN/Infinity with 0 (matches np.nan_to_num in training)
function sanitizeArray(arr: Float32Array): Float32Array {
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isFinite(arr[i])) {
      arr[i] = 0;
    }
  }
  return arr;
}

// Generate a simple NDVI-based segmentation mask as PNG data URL
function generateNdviMask(
  redData: Float32Array,
  nirData: Float32Array,
  width: number,
  height: number
): { maskDataUrl: string; forestPercentage: number } {
  const ndvi = new Float32Array(width * height);
  let forestPixels = 0;
  
  for (let i = 0; i < width * height; i++) {
    const red = redData[i];
    const nir = nirData[i];
    
    if (nir + red > 0) {
      ndvi[i] = (nir - red) / (nir + red);
    } else {
      ndvi[i] = 0;
    }
    
    // NDVI > 0.3 is typically vegetation/forest
    if (ndvi[i] > 0.3) {
      forestPixels++;
    }
  }
  
  const rgba = new Uint8Array(width * height * 4);
  
  for (let i = 0; i < width * height; i++) {
    const idx = i * 4;
    const val = ndvi[i];
    
    if (val > 0.6) {
      rgba[idx] = 0;
      rgba[idx + 1] = 128;
      rgba[idx + 2] = 0;
      rgba[idx + 3] = 200;
    } else if (val > 0.4) {
      rgba[idx] = 34;
      rgba[idx + 1] = 139;
      rgba[idx + 2] = 34;
      rgba[idx + 3] = 180;
    } else if (val > 0.2) {
      rgba[idx] = 144;
      rgba[idx + 1] = 238;
      rgba[idx + 2] = 144;
      rgba[idx + 3] = 150;
    } else if (val > 0) {
      rgba[idx] = 210;
      rgba[idx + 1] = 180;
      rgba[idx + 2] = 140;
      rgba[idx + 3] = 120;
    } else {
      rgba[idx] = 65;
      rgba[idx + 1] = 105;
      rgba[idx + 2] = 225;
      rgba[idx + 3] = val < -0.1 ? 150 : 50;
    }
  }
  
  const bmpData = createBMP(rgba, width, height);
  const base64 = uint8ToBase64(bmpData);

  return {
    maskDataUrl: `data:image/bmp;base64,${base64}`,
    forestPercentage: (forestPixels / (width * height)) * 100
  };
}

// Create a simple BMP file from RGBA data
function createBMP(rgba: Uint8Array, width: number, height: number): Uint8Array {
  const rowSize = Math.ceil((width * 4) / 4) * 4;
  const imageSize = rowSize * height;
  const fileSize = 54 + imageSize;
  
  const bmp = new Uint8Array(fileSize);
  const view = new DataView(bmp.buffer);
  
  bmp[0] = 0x42;
  bmp[1] = 0x4D;
  view.setUint32(2, fileSize, true);
  view.setUint32(10, 54, true);
  
  view.setUint32(14, 40, true);
  view.setInt32(18, width, true);
  view.setInt32(22, -height, true);
  view.setUint16(26, 1, true);
  view.setUint16(28, 32, true);
  view.setUint32(30, 0, true);
  view.setUint32(34, imageSize, true);
  view.setUint32(38, 2835, true);
  view.setUint32(42, 2835, true);
  
  let offset = 54;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const srcIdx = (y * width + x) * 4;
      bmp[offset++] = rgba[srcIdx + 2];
      bmp[offset++] = rgba[srcIdx + 1];
      bmp[offset++] = rgba[srcIdx];
      bmp[offset++] = rgba[srcIdx + 3];
    }
  }
  
  return bmp;
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const ML_MODEL_URL = Deno.env.get('ML_MODEL_URL');
    
    const body: PredictRequest = await req.json();
    const { 
      bbox, 
      startDate: rawStartDate,
      endDate: rawEndDate,
      modelName = 'forest_segmentation',
      modelVersion = 'v1',
      forceNdvi = false,
      threshold: userThreshold = 128
    } = body;
    
    console.log(`User threshold: ${userThreshold}`);
    
    const mlConfigured = !!ML_MODEL_URL;
    const useDemoMode = !mlConfigured || forceNdvi;
    
    console.log(`ML Model URL configured: ${mlConfigured}, forceNdvi: ${forceNdvi}, using demo mode: ${useDemoMode}`);

    const formatDate = (dateStr: string | undefined, isEnd: boolean): string => {
      if (!dateStr) {
        const d = isEnd ? new Date() : new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
        return d.toISOString();
      }
      if (dateStr.includes('T')) return dateStr;
      return isEnd ? `${dateStr}T23:59:59Z` : `${dateStr}T00:00:00Z`;
    };

    const startDate = formatDate(rawStartDate, false);
    const endDate = formatDate(rawEndDate, true);

    if (!bbox) {
      return new Response(
        JSON.stringify({ error: 'bbox is required' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    console.log('=== LANDSAT 8 MODE ===');
    console.log('Fetching Landsat 8 L2 bands for bbox:', bbox);
    console.log('Date range:', startDate, 'to', endDate);

    const { accessToken, processUrl, provider } = await getSentinelToken();
    console.log(`Got Sentinel Hub token (provider=${provider})`);

    // Fetch all 6 Landsat 8 optical bands
    const bandsToFetch = Object.entries(LANDSAT8_BANDS);
    const bboxArray = [bbox.minLng, bbox.minLat, bbox.maxLng, bbox.maxLat];

    const width = 256;
    const height = 256;

    console.log('Fetching Landsat 8 bands:', bandsToFetch.map(([name, band]) => `${name}=${band}`).join(', '));
    
    const bandDataPromises = bandsToFetch.map(([_name, band]) =>
      fetchLandsat8BandData(accessToken, processUrl, bboxArray, band, startDate, endDate, width, height)
    );

    const bandDataArrays = await Promise.all(bandDataPromises);
    console.log('All Landsat 8 bands fetched (raw reflectance 0-1)');

    // Store raw reflectance for index calculation
    const rawReflectance: Record<string, Float32Array> = {};
    const opticalBandNames = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2'];
    
    for (let i = 0; i < bandsToFetch.length; i++) {
      const [name] = bandsToFetch[i];
      rawReflectance[name] = sanitizeArray(bandDataArrays[i]);
    }

    // STEP 1: Calculate spectral indices on RAW reflectance (0-1 range)
    // This matches GEE where indices are calculated on reflectance values
    // BEFORE the -0.2 offset is applied to the optical bands
    console.log('Calculating spectral indices (NDVI, NDWI, NBR) on raw reflectance...');
    const ndvi = calculateNDVI(rawReflectance['NIR'], rawReflectance['Red']);
    const ndwi = calculateNDWI(rawReflectance['Green'], rawReflectance['NIR']);
    const nbr = calculateNBR(rawReflectance['NIR'], rawReflectance['SWIR2']);

    sanitizeArray(ndvi);
    sanitizeArray(ndwi);
    sanitizeArray(nbr);

    // STEP 2: Apply GEE offset (-0.2) to optical bands only
    // This matches GEE's applyScaleFactors: multiply(0.0000275).add(-0.2)
    // Since Sentinel Hub gives us reflectance (0-1), we just need to subtract 0.2
    const bandDataMap: Record<string, Float32Array> = {};
    
    // Apply GEE offset and CLIP to training data range [-0.2, 0.6]
    for (const name of opticalBandNames) {
      const raw = rawReflectance[name];
      const scaled = new Float32Array(raw.length);
      for (let j = 0; j < raw.length; j++) {
        const withOffset = raw[j] + GEE_OFFSET; // Apply -0.2 offset
        // CLIP to GEE training range to avoid out-of-distribution values
        scaled[j] = Math.max(-0.2, Math.min(0.6, withOffset));
      }
      bandDataMap[name] = scaled;
    }
    
    console.log('Applied GEE offset (-0.2) and clipped optical bands to [-0.2, 0.6]');

    // CLIP indices to valid range [-1, 1]
    for (let j = 0; j < ndvi.length; j++) {
      ndvi[j] = Math.max(-1.0, Math.min(1.0, ndvi[j]));
      ndwi[j] = Math.max(-1.0, Math.min(1.0, ndwi[j]));
      nbr[j] = Math.max(-1.0, Math.min(1.0, nbr[j]));
    }
    console.log('Clipped spectral indices to [-1.0, 1.0]');

    // Add indices
    bandDataMap['NDVI'] = ndvi;
    bandDataMap['NDWI'] = ndwi;
    bandDataMap['NBR'] = nbr;

    // The 9-channel stack order matching your GEE training:
    // [Blue, Green, Red, NIR, SWIR1, SWIR2, NDVI, NDWI, NBR]
    const channelOrder = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDWI', 'NBR'];

    // Logging for verification
    const summarize = (arr: Float32Array) => {
      let min = Infinity;
      let max = -Infinity;
      let sum = 0;
      let nonZero = 0;
      for (let i = 0; i < arr.length; i++) {
        const v = arr[i];
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
        if (v !== 0) nonZero++;
      }
      return {
        min: Number.isFinite(min) ? min : 0,
        max: Number.isFinite(max) ? max : 0,
        mean: sum / (arr.length || 1),
        nonZeroPct: (nonZero / (arr.length || 1)) * 100,
      };
    };

    console.log('=== 9-Channel Input Statistics (GEE-Scaled) ===');
    console.log('Expected ranges: Optical bands ~[-0.2, 0.6], Indices ~[-1, 1]');
    for (const name of channelOrder) {
      const s = summarize(bandDataMap[name]);
      console.log(`${name}: min=${s.min.toFixed(6)} max=${s.max.toFixed(6)} mean=${s.mean.toFixed(6)} nonZero%=${s.nonZeroPct.toFixed(2)}`);
    }

    // Demo mode: NDVI-based mask
    if (useDemoMode) {
      console.log('Generating NDVI-based demo mask');
      const { maskDataUrl, forestPercentage } = generateNdviMask(
        bandDataMap['Red'],
        bandDataMap['NIR'],
        width,
        height
      );

      return new Response(
        JSON.stringify({
          success: true,
          prediction: {
            mask: maskDataUrl,
            forest_percentage: Math.round(forestPercentage * 10) / 10,
            confidence: 0.75,
            classes: ['Dense Forest', 'Moderate Vegetation', 'Sparse Vegetation', 'Bare Soil', 'Water'],
            demo_mode: true,
          },
          bbox,
          modelName: 'ndvi_demo',
          modelVersion: 'v1',
          ml_available: mlConfigured,
          width,
          height
        }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Prepare ML model payload with 9 channels
    // Python model expects band names: Blue, Green, Red, NIR, SWIR1, SWIR2, NDVI, NDWI, NBR
    const bandsBase64: Record<string, string> = {};
    for (const name of channelOrder) {
      bandsBase64[name] = float32ArrayToBase64(bandDataMap[name]);
    }

    const mlPayload = {
      model_name: modelName,
      model_version: modelVersion,
      bands: bandsBase64,
      width,
      height,
      bbox,
      band_names: channelOrder,
      preprocessing: {
        dtype: 'float32',
        layout: 'hwc',
        shape: [height, width, channelOrder.length],
        scaling: 'gee_landsat8',
        gee_formula: 'multiply(0.0000275).add(-0.2)',
      },
    };

    console.log('Sending ML bands (base64 float32):', channelOrder);
    console.log('Example band payload length (Blue):', bandsBase64['Blue']?.length ?? 0);
    console.log('Sending to ML model:', ML_MODEL_URL);

    // Add timeout with AbortController (60 seconds for ML inference)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000);

    let mlResponse: Response;
    try {
      mlResponse = await fetch(ML_MODEL_URL!, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(mlPayload),
        signal: controller.signal,
      });
    } catch (fetchError) {
      clearTimeout(timeoutId);
      const errorMessage = fetchError instanceof Error ? fetchError.message : 'Unknown fetch error';
      console.error('ML model fetch error:', errorMessage);
      
      // Fallback to NDVI mode on connection error
      console.log('Falling back to NDVI mode due to ML connection error');
      const { maskDataUrl, forestPercentage } = generateNdviMask(bandDataMap['Red'], bandDataMap['NIR'], 256, 256);
      return new Response(
        JSON.stringify({
          success: true,
          prediction: {
            mask: maskDataUrl,
            forest_percentage: forestPercentage,
            confidence: 0.7,
            classes: ['forest', 'non_forest'],
            demo_mode: true,
          },
          bbox,
          modelType: 'ndvi_fallback',
          ml_available: false,
          message: `ML model unavailable (${errorMessage}), using NDVI fallback`,
        }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }
    clearTimeout(timeoutId);

    if (!mlResponse.ok) {
      const errorText = await mlResponse.text();
      console.error('ML model error:', mlResponse.status, errorText);
      return new Response(
        JSON.stringify({ 
          error: 'ML model prediction failed',
          status: mlResponse.status,
          details: errorText
        }),
        { status: mlResponse.status, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const result = await mlResponse.json();

    console.log('ML response keys:', JSON.stringify(result ? Object.keys(result) : 'no_result'));
    
    // Check what mask fields are available and their types
    const hasInvertedMask = result && result.inverted_mask !== undefined && result.inverted_mask !== null;
    const hasMask = result && result.mask !== undefined && result.mask !== null;
    
    console.log('Mask availability:', { 
      inverted_mask: hasInvertedMask ? (Array.isArray(result.inverted_mask) ? `array[${result.inverted_mask.length}]` : typeof result.inverted_mask) : 'missing',
      mask: hasMask ? (Array.isArray(result.mask) ? `array[${result.mask.length}]` : typeof result.mask) : 'missing'
    });
    
    // User's local inference uses 'mask' (channel 0) where HIGH = FOREST
    // Prioritize 'mask' field to match local script behavior
    let rawMask: unknown;
    let usedMaskField: string;
    
    if (hasMask) {
      // Primary: use 'mask' field (channel 0) - matches local inference
      rawMask = result.mask;
      usedMaskField = 'mask';
    } else if (hasInvertedMask) {
      // Fallback to inverted_mask if mask not available
      rawMask = result.inverted_mask;
      usedMaskField = 'inverted_mask';
    } else if (result?.prediction_mask) {
      rawMask = result.prediction_mask;
      usedMaskField = 'prediction_mask';
    } else if (result?.mask_base64) {
      rawMask = result.mask_base64;
      usedMaskField = 'mask_base64';
    } else {
      rawMask = undefined;
      usedMaskField = 'none';
    }
    
    console.log('Using mask field:', usedMaskField);
    if (rawMask && typeof rawMask === 'object' && !Array.isArray(rawMask)) {
      const maskObj = rawMask as Record<string, unknown>;
      console.log('ML mask object keys:', Object.keys(maskObj));
    } else if (Array.isArray(rawMask)) {
      console.log(`ML mask is direct array: length=${rawMask.length}, first=${Array.isArray(rawMask[0]) ? 'nested' : 'flat'}`);
    } else {
      console.log('ML mask type:', typeof rawMask, 'raw length:', typeof rawMask === 'string' ? (rawMask as string).length : -1);
    }

    const flatten2D = (arr: unknown[]): number[] => {
      if (!arr.length) return [];
      if (Array.isArray(arr[0])) return (arr as unknown[][]).flat(Infinity).map((v) => Number(v));
      return arr.map((v) => Number(v));
    };

    // Auto-inversion detection: If mask mean > 0.7 (on 0-1 scale) or > 178 (on 0-255 scale),
    // the model likely outputs high=non-forest, so we invert to high=forest
    const detectAndCorrectInversion = (flat: number[], ndviData?: Float32Array): { corrected: number[]; wasInverted: boolean; meanBefore: number } => {
      // Calculate statistics
      let sum = 0;
      let count = 0;
      let max = -Infinity;
      let min = Infinity;
      let nonZero = 0;
      
      for (const v of flat) {
        if (Number.isFinite(v)) {
          sum += v;
          count++;
          if (v > max) max = v;
          if (v < min) min = v;
          if (v !== 0) nonZero++;
        }
      }
      
      const mean = count > 0 ? sum / count : 0;
      const scale = max > 1 ? 255 : 1; // Detect if 0-255 or 0-1 scale
      const normalizedMean = scale > 1 ? mean / 255 : mean;
      
      console.log(`[INVERSION CHECK] Mean: ${mean.toFixed(4)} (normalized: ${normalizedMean.toFixed(4)}), Min: ${min.toFixed(4)}, Max: ${max}, NonZero: ${nonZero}/${count}, Scale: ${scale}`);

      // Guard: if the model returned an empty/constant mask (all zeros or no variance),
      // inversion logic would turn it into an all-ones mask (100% forest) which is misleading.
      const hasVariance = Number.isFinite(min) && Number.isFinite(max) && max !== min;
      if (count === 0 || max <= 0 || nonZero === 0 || !hasVariance) {
        console.log(`[INVERSION] Skipping inversion: empty/constant mask (min=${min}, max=${max}, nonZero=${nonZero}).`);
        return { corrected: flat, wasInverted: false, meanBefore: mean };
      }
      
      // User's local script uses 'mask' (channel 0) where HIGH = FOREST
      // Trust the model output directly - no inversion needed for 'mask' field
      // Only invert for 'inverted_mask' field (if that's where high = non-forest)
      let shouldInvert = false;
      
      if (usedMaskField === 'inverted_mask') {
        // 'inverted_mask' may have opposite semantics - check if inversion needed
        // But typically inverted_mask should already be correct, skip inversion
        console.log(`[INVERSION] Using 'inverted_mask' field - trusting model output`);
        shouldInvert = false;
      } else {
        // 'mask' field: user confirmed high = forest, no inversion needed
        console.log(`[INVERSION] Using '${usedMaskField}' field - high values = FOREST (no inversion)`);
        shouldInvert = false;
      }
      
      if (shouldInvert) {
        console.log(`[INVERSION] Inverting mask values: 255 - value (or 1 - value)`);
        const corrected = flat.map(v => {
          if (!Number.isFinite(v)) return 0;
          return scale > 1 ? 255 - v : 1 - v;
        });
        return { corrected, wasInverted: true, meanBefore: mean };
      }
      
      return { corrected: flat, wasInverted: false, meanBefore: mean };
    };

    const arrayToBmp = (flat: number[], w: number, h: number, ndviData?: Float32Array, customThreshold?: number): string => {
      // Apply auto-inversion detection
      const { corrected, wasInverted, meanBefore } = detectAndCorrectInversion(flat, ndviData);
      
      if (wasInverted) {
        const newMean = corrected.reduce((a, b) => a + b, 0) / corrected.length;
        console.log(`[INVERSION] Applied! Mean before: ${meanBefore.toFixed(4)}, after: ${newMean.toFixed(4)}`);
      }
      
      const rgba = new Uint8Array(w * h * 4);
      let min = Infinity, max = -Infinity;
      for (const v of corrected) {
        if (Number.isFinite(v)) {
          if (v < min) min = v;
          if (v > max) max = v;
        }
      }
      console.log(`Mask value range (after correction): min=${min}, max=${max}`);

      // Determine scale (0-255 or 0-1) and apply user threshold
      // User threshold from UI is on 0-255 scale, convert to probability if needed
      const scale = max > 1 ? 255 : 1;
      let threshold: number;
      if (customThreshold !== undefined) {
        // Convert UI threshold (0-255) to the appropriate scale
        threshold = scale > 1 ? customThreshold : customThreshold / 255;
      } else {
        // Default: 0.5 probability (matches local script) = 128 on 0-255 scale
        threshold = scale > 1 ? 128 : 0.5;
      }
      console.log(`Using threshold: ${threshold} (scale: ${scale}, user provided: ${customThreshold})`);

      // Binary color mask:
      // - HIGH values (>= threshold) = Forest = GREEN
      // - LOW values (< threshold) = Non-forest = BLUE
      for (let i = 0; i < w * h; i++) {
        const rawVal = Number.isFinite(corrected[i]) ? corrected[i] : 0;
        const idx = i * 4;
        
        if (rawVal >= threshold) {
          // Forest = GREEN (#22c55e)
          rgba[idx] = 34;      // R
          rgba[idx + 1] = 197; // G
          rgba[idx + 2] = 94;  // B
          rgba[idx + 3] = 255; // A (fully opaque)
        } else {
          // Non-forest = BLUE (#3b82f6)
          rgba[idx] = 59;      // R
          rgba[idx + 1] = 130; // G
          rgba[idx + 2] = 246; // B
          rgba[idx + 3] = 255; // A (fully opaque)
        }
      }
      const bmpData = createBMP(rgba, w, h);
      return `data:image/bmp;base64,${uint8ToBase64(bmpData)}`;
    };

    // Pass NDVI data for correlation-based inversion detection
    const ndviForCorrelation = bandDataMap['NDVI'];
    
    const normalizeMaskToDataUrl = (m: unknown): string | null => {
      if (typeof m === 'string') {
        const trimmed = m.trim();
        if (!trimmed) return null;
        if (trimmed.startsWith('data:image/')) return trimmed;
        if (/^[A-Za-z0-9+/=]+$/.test(trimmed) && trimmed.length > 50) {
          return `data:image/png;base64,${trimmed}`;
        }
        return null;
      }

      if (Array.isArray(m)) {
        const flat = flatten2D(m);
        console.log(`Direct array mask: length=${flat.length}, expected=${width * height}`);
        
        const nonZeroCount = flat.filter(v => v !== 0).length;
        const sampleVals = flat.slice(0, 20).map(v => v.toFixed(4)).join(', ');
        console.log(`Mask sample (first 20 values): [${sampleVals}]`);
        console.log(`Mask non-zero count: ${nonZeroCount} / ${flat.length} (${((nonZeroCount / flat.length) * 100).toFixed(2)}%)`);
        
        if (flat.length === width * height) {
          return arrayToBmp(flat, width, height, ndviForCorrelation, userThreshold);
        }
        const sqrt = Math.sqrt(flat.length);
        if (Number.isInteger(sqrt) && sqrt >= 64) {
          console.log(`Inferred square mask: ${sqrt}x${sqrt}`);
          return arrayToBmp(flat, sqrt, sqrt, ndviForCorrelation, userThreshold);
        }
      }

      if (m && typeof m === 'object') {
        const obj = m as Record<string, unknown>;
        
        const stringKeys = ['data', 'base64', 'png_base64', 'mask_base64', 'image', 'png', 'b64'];
        for (const key of stringKeys) {
          if (typeof obj[key] === 'string') {
            const normalized = normalizeMaskToDataUrl(obj[key]);
            if (normalized) return normalized;
          }
        }

        const arrayKeys = ['values', 'array', 'mask', 'predictions', 'output', 'segmentation'];
        for (const key of arrayKeys) {
          if (Array.isArray(obj[key])) {
            const flat = flatten2D(obj[key] as unknown[]);
            console.log(`Object mask.${key}: flat length=${flat.length}, expected=${width * height}`);
            if (flat.length === width * height) {
              return arrayToBmp(flat, width, height, ndviForCorrelation, userThreshold);
            }
          }
        }
      }

      return null;
    };

    const normalizedMask = normalizeMaskToDataUrl(rawMask);
    console.log(`Normalized mask: ${normalizedMask?.slice(0, 60)}... (len=${normalizedMask?.length ?? 0})`);

    const forestPct: number | undefined =
      result?.forest_percentage ?? result?.forestPercentage ?? result?.prediction?.forest_percentage;
    const confidenceVal: number | undefined =
      result?.confidence ?? result?.prediction?.confidence;
    const classesArr: string[] | undefined =
      result?.classes ?? result?.prediction?.classes;
    const lossAreas: unknown[] | undefined =
      result?.loss_areas ?? result?.prediction?.loss_areas;
    const demoModeFlag: boolean =
      result?.demo_mode ?? result?.prediction?.demo_mode ?? false;

    console.log('ML prediction successful');

    return new Response(
      JSON.stringify({
        success: true,
        prediction: {
          mask: normalizedMask,
          forest_percentage: forestPct,
          confidence: confidenceVal,
          classes: classesArr,
          loss_areas: lossAreas,
          demo_mode: demoModeFlag,
        },
        bbox,
        modelType: 'segmentation',
        ml_available: true,
        width,
        height,
        preprocessing: 'landsat8_gee_matched',
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error('Prediction error:', error);
    return new Response(
      JSON.stringify({ 
        error: 'Prediction failed',
        message: error instanceof Error ? error.message : String(error)
      }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});
