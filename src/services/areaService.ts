import { supabase } from "@/integrations/supabase/client";
import type { BoundingBox } from "@/types/aoi";
import type { Json } from "@/integrations/supabase/types";

export interface AreaPayload {
  type: "Polygon";
  coordinates: [number, number][];
  boundingBox?: BoundingBox;
}

export interface AreaResponse {
  success: boolean;
  message: string;
  data?: {
    id: string;
    area_type: string;
    coordinates: [number, number][];
    bounding_box: BoundingBox | null;
    created_at: string;
  };
}

/**
 * Save selected area to Lovable Cloud database
 */
export async function saveArea(payload: AreaPayload): Promise<AreaResponse> {
  const { data, error } = await supabase
    .from('saved_areas')
    .insert([{
      area_type: payload.type,
      coordinates: payload.coordinates as Json,
      bounding_box: payload.boundingBox ? (payload.boundingBox as unknown as Json) : null,
    }])
    .select()
    .single();

  if (error) {
    throw new Error(`Failed to save area: ${error.message}`);
  }

  return {
    success: true,
    message: 'Area saved successfully',
    data: {
      id: data.id,
      area_type: data.area_type,
      coordinates: data.coordinates as unknown as [number, number][],
      bounding_box: data.bounding_box as unknown as BoundingBox | null,
      created_at: data.created_at,
    },
  };
}

/**
 * Fetch the latest saved area from database
 */
export async function getLatestArea(): Promise<AreaResponse> {
  const { data, error } = await supabase
    .from('saved_areas')
    .select('*')
    .order('created_at', { ascending: false })
    .limit(1)
    .maybeSingle();

  if (error) {
    throw new Error(`Failed to fetch area: ${error.message}`);
  }

  if (!data) {
    return {
      success: false,
      message: 'No saved areas found',
    };
  }

  return {
    success: true,
    message: 'Area retrieved successfully',
    data: {
      id: data.id,
      area_type: data.area_type,
      coordinates: data.coordinates as unknown as [number, number][],
      bounding_box: data.bounding_box as unknown as BoundingBox | null,
      created_at: data.created_at,
    },
  };
}
