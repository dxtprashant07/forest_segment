import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

type VisualizationType = "true_color" | "ndvi" | "ndwi" | "false_color" | "swir" | "nbr";

interface FetchSatelliteRequest {
  bbox: {
    minLat: number;
    maxLat: number;
    minLng: number;
    maxLng: number;
  };
  startDate?: string;
  endDate?: string;
  resolution?: number;
  visualization?: VisualizationType;
}

// ============= LANDSAT 8 BAND MAPPING =============
// Landsat 8 Collection 2 Level-2 bands:
// B02 = Blue, B03 = Green, B04 = Red
// B05 = NIR, B06 = SWIR1, B07 = SWIR2

// Evalscripts for different visualization types using Landsat 8 bands
const getEvalscript = (visualization: VisualizationType): string => {
  switch (visualization) {
    case "ndvi":
      // NDVI = (NIR - Red) / (NIR + Red) = (B05 - B04) / (B05 + B04)
      return `
        //VERSION=3
        function setup() {
          return {
            input: ["B04", "B05"],
            output: { bands: 3 }
          };
        }
        
        function evaluatePixel(sample) {
          let ndvi = (sample.B05 - sample.B04) / (sample.B05 + sample.B04);
          
          // Handle no-data
          if (isNaN(ndvi) || !isFinite(ndvi)) {
            return [0.5, 0.5, 0.5];
          }
          
          // Color ramp: Red (low/negative) -> Yellow -> Green (high/positive)
          if (ndvi < -0.2) {
            // Water/bare - deep blue
            return [0.05, 0.1, 0.4];
          } else if (ndvi < 0) {
            // Bare soil/urban - brown
            return [0.5, 0.35, 0.2];
          } else if (ndvi < 0.1) {
            // Sparse vegetation - tan
            return [0.75, 0.65, 0.45];
          } else if (ndvi < 0.2) {
            // Low vegetation - yellow-green
            return [0.8, 0.8, 0.3];
          } else if (ndvi < 0.4) {
            // Moderate vegetation - light green
            return [0.55, 0.75, 0.25];
          } else if (ndvi < 0.6) {
            // Good vegetation - green
            return [0.25, 0.65, 0.15];
          } else {
            // Dense vegetation - dark green
            return [0.1, 0.45, 0.1];
          }
        }
      `;

    case "ndwi":
      // NDWI = (Green - NIR) / (Green + NIR) = (B03 - B05) / (B03 + B05)
      return `
        //VERSION=3
        function setup() {
          return {
            input: ["B03", "B05"],
            output: { bands: 3 }
          };
        }
        
        function evaluatePixel(sample) {
          let ndwi = (sample.B03 - sample.B05) / (sample.B03 + sample.B05);
          
          if (isNaN(ndwi) || !isFinite(ndwi)) {
            return [0.5, 0.5, 0.5];
          }
          
          // Color ramp: Brown (dry) -> Blue (water)
          if (ndwi < -0.3) {
            // Very dry/bare - brown
            return [0.55, 0.35, 0.2];
          } else if (ndwi < -0.1) {
            // Dry vegetation - tan
            return [0.7, 0.55, 0.35];
          } else if (ndwi < 0) {
            // Moist soil - grayish
            return [0.6, 0.6, 0.55];
          } else if (ndwi < 0.2) {
            // Wet/moist - light blue
            return [0.5, 0.7, 0.85];
          } else if (ndwi < 0.4) {
            // Water/wetland - medium blue
            return [0.3, 0.55, 0.85];
          } else {
            // Deep water - dark blue
            return [0.1, 0.3, 0.7];
          }
        }
      `;

    case "nbr":
      // NBR = (NIR - SWIR2) / (NIR + SWIR2) = (B05 - B07) / (B05 + B07)
      return `
        //VERSION=3
        function setup() {
          return {
            input: ["B05", "B07"],
            output: { bands: 3 }
          };
        }
        
        function evaluatePixel(sample) {
          let nbr = (sample.B05 - sample.B07) / (sample.B05 + sample.B07);
          
          if (isNaN(nbr) || !isFinite(nbr)) {
            return [0.5, 0.5, 0.5];
          }
          
          // Color ramp: Dark red (burned) -> Orange -> Yellow -> Green (healthy)
          if (nbr < -0.5) {
            // Severely burned
            return [0.3, 0.05, 0.05];
          } else if (nbr < -0.25) {
            // Moderately burned
            return [0.6, 0.15, 0.05];
          } else if (nbr < -0.1) {
            // Low burn
            return [0.8, 0.35, 0.1];
          } else if (nbr < 0.1) {
            // Unburned bare
            return [0.9, 0.7, 0.4];
          } else if (nbr < 0.3) {
            // Low vegetation
            return [0.7, 0.8, 0.4];
          } else if (nbr < 0.5) {
            // Moderate vegetation
            return [0.4, 0.7, 0.25];
          } else {
            // Healthy vegetation
            return [0.15, 0.55, 0.15];
          }
        }
      `;

    case "false_color":
      // NIR-Red-Green composite (vegetation appears bright red)
      // Landsat 8: B05 (NIR), B04 (Red), B03 (Green)
      return `
        //VERSION=3
        function setup() {
          return {
            input: ["B03", "B04", "B05"],
            output: { bands: 3 }
          };
        }
        function evaluatePixel(sample) {
          return [2.5 * sample.B05, 2.5 * sample.B04, 2.5 * sample.B03];
        }
      `;

    case "swir":
      // SWIR composite for geology/moisture
      // Landsat 8: B07 (SWIR2), B05 (NIR), B04 (Red)
      return `
        //VERSION=3
        function setup() {
          return {
            input: ["B04", "B05", "B07"],
            output: { bands: 3 }
          };
        }
        function evaluatePixel(sample) {
          return [2.5 * sample.B07, 2.5 * sample.B05, 2.5 * sample.B04];
        }
      `;

    case "true_color":
    default:
      // True color: Red, Green, Blue
      // Landsat 8: B04 (Red), B03 (Green), B02 (Blue)
      return `
        //VERSION=3
        function setup() {
          return {
            input: ["B02", "B03", "B04"],
            output: { bands: 3 }
          };
        }
        function evaluatePixel(sample) {
          return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
        }
      `;
  }
};

serve(async (req) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const SENTINEL_CLIENT_ID = Deno.env.get('SENTINEL_CLIENT_ID');
    const SENTINEL_CLIENT_SECRET = Deno.env.get('SENTINEL_CLIENT_SECRET');

    if (!SENTINEL_CLIENT_ID || !SENTINEL_CLIENT_SECRET) {
      console.log('Sentinel Hub credentials not configured');
      return new Response(
        JSON.stringify({ 
          error: 'Sentinel Hub API not configured',
          message: 'Please add SENTINEL_CLIENT_ID and SENTINEL_CLIENT_SECRET to your secrets'
        }),
        { 
          status: 503, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      );
    }

    const body: FetchSatelliteRequest = await req.json();
    const { bbox, startDate, endDate, resolution = 30, visualization = 'true_color' } = body;

    console.log('Fetching satellite imagery for bbox:', bbox, 'visualization:', visualization);

    // Get access token
    const clientId = SENTINEL_CLIENT_ID.trim();
    const clientSecret = SENTINEL_CLIENT_SECRET.trim();
    
    const tokenResponse = await fetch('https://services.sentinel-hub.com/oauth/token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        grant_type: 'client_credentials',
        client_id: clientId,
        client_secret: clientSecret,
      }),
    });

    if (!tokenResponse.ok) {
      const errorText = await tokenResponse.text();
      console.error('Token fetch failed:', errorText);
      return new Response(
        JSON.stringify({ 
          error: 'Failed to authenticate with Sentinel Hub', 
          details: errorText,
        }),
        { status: 401, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const tokenData = await tokenResponse.json();
    const accessToken = tokenData.access_token;

    // Calculate date range (default: last 30 days)
    const end = endDate ? new Date(endDate) : new Date();
    const start = startDate ? new Date(startDate) : new Date(end.getTime() - 30 * 24 * 60 * 60 * 1000);

    // Get the appropriate evalscript for the visualization type
    const evalscript = getEvalscript(visualization);

    // Calculate image dimensions based on bbox and resolution
    const latDiff = bbox.maxLat - bbox.minLat;
    const lngDiff = bbox.maxLng - bbox.minLng;
    const avgLat = (bbox.maxLat + bbox.minLat) / 2;
    const metersPerDegreeLat = 111320;
    const metersPerDegreeLng = 111320 * Math.cos(avgLat * Math.PI / 180);
    
    const heightMeters = latDiff * metersPerDegreeLat;
    const widthMeters = lngDiff * metersPerDegreeLng;
    
    // Calculate pixel dimensions (capped at 2500px to avoid API limits)
    const width = Math.min(Math.round(widthMeters / resolution), 2500);
    const height = Math.min(Math.round(heightMeters / resolution), 2500);

    const processRequest = {
      input: {
        bounds: {
          bbox: [bbox.minLng, bbox.minLat, bbox.maxLng, bbox.maxLat],
          properties: {
            crs: 'http://www.opengis.net/def/crs/EPSG/0/4326'
          }
        },
        data: [{
          type: 'landsat-ot-l2',  // Landsat 8-9 Level 2 Collection
          dataFilter: {
            timeRange: {
              from: start.toISOString(),
              to: end.toISOString(),
            },
            maxCloudCoverage: 30,
          },
        }],
      },
      output: {
        width: Math.max(width, 256),
        height: Math.max(height, 256),
        responses: [{
          identifier: 'default',
          format: { type: 'image/png' }
        }],
      },
      evalscript,
    };

    console.log(`Requesting Landsat 8 ${visualization} imagery (${width}x${height}px at ${resolution}m resolution)...`);

    // Use US-West-2 endpoint which hosts Landsat 8-9 L2 data
    const imageResponse = await fetch('https://services-uswest2.sentinel-hub.com/api/v1/process', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${accessToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(processRequest),
    });

    if (!imageResponse.ok) {
      const errorText = await imageResponse.text();
      console.error('Image fetch failed:', imageResponse.status, errorText);
      return new Response(
        JSON.stringify({ 
          error: 'Failed to fetch satellite imagery', 
          status: imageResponse.status,
          details: errorText 
        }),
        { status: imageResponse.status, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Return the image as base64
    const imageBuffer = await imageResponse.arrayBuffer();
    const bytes = new Uint8Array(imageBuffer);
    let binary = '';
    const chunkSize = 8192;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      const chunk = bytes.subarray(i, i + chunkSize);
      binary += String.fromCharCode(...chunk);
    }
    const base64Image = btoa(binary);

    console.log('Successfully fetched Landsat 8', visualization, 'imagery');

    return new Response(
      JSON.stringify({ 
        success: true,
        image: `data:image/png;base64,${base64Image}`,
        bbox,
        visualization,
        resolution,
        dateRange: {
          start: start.toISOString(),
          end: end.toISOString(),
        }
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error('Error in fetch-satellite function:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    return new Response(
      JSON.stringify({ error: errorMessage }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});
