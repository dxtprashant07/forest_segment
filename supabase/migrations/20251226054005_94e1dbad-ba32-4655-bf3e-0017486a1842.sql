-- Create table for storing saved areas
CREATE TABLE public.saved_areas (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  area_type TEXT NOT NULL DEFAULT 'Polygon',
  coordinates JSONB NOT NULL,
  bounding_box JSONB,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable RLS but allow public access for now (no auth required)
ALTER TABLE public.saved_areas ENABLE ROW LEVEL SECURITY;

-- Allow anyone to insert and read areas (public feature)
CREATE POLICY "Anyone can insert areas" 
ON public.saved_areas 
FOR INSERT 
WITH CHECK (true);

CREATE POLICY "Anyone can read areas" 
ON public.saved_areas 
FOR SELECT 
USING (true);