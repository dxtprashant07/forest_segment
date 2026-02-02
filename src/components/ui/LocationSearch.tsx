import { useState, useCallback } from "react";
import { Search, MapPin, Loader2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useLocationSearch } from "@/hooks/useLocationSearch";
import { cn } from "@/lib/utils";

interface LocationSearchProps {
  onLocationSelect: (lat: number, lng: number, zoom: number) => void;
}

export function LocationSearch({ onLocationSelect }: LocationSearchProps) {
  const [query, setQuery] = useState("");
  const [showResults, setShowResults] = useState(false);
  const { searchResults, isSearching, searchError, searchLocation, clearSearch } = useLocationSearch();

  const handleSearch = useCallback(() => {
    if (query.trim()) {
      searchLocation(query);
      setShowResults(true);
    }
  }, [query, searchLocation]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  const handleResultClick = (lat: string, lon: string) => {
    onLocationSelect(parseFloat(lat), parseFloat(lon), 12);
    setShowResults(false);
    clearSearch();
  };

  return (
    <div className="relative">
      <div className="flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search location..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => searchResults.length > 0 && setShowResults(true)}
            className="pl-10"
          />
        </div>
        <Button onClick={handleSearch} disabled={isSearching || !query.trim()}>
          {isSearching ? <Loader2 className="h-4 w-4 animate-spin" /> : "Search"}
        </Button>
      </div>

      {/* Search Results Dropdown */}
      {showResults && searchResults.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-popover border border-border rounded-md shadow-lg z-50 max-h-64 overflow-y-auto">
          {searchResults.map((result, index) => (
            <button
              key={index}
              onClick={() => handleResultClick(result.lat, result.lon)}
              className={cn(
                "w-full px-4 py-3 text-left flex items-start gap-3 hover:bg-accent transition-colors",
                index !== searchResults.length - 1 && "border-b border-border"
              )}
            >
              <MapPin className="h-4 w-4 mt-0.5 text-muted-foreground shrink-0" />
              <span className="text-sm text-foreground line-clamp-2">{result.display_name}</span>
            </button>
          ))}
        </div>
      )}

      {/* Error Message */}
      {searchError && (
        <p className="text-sm text-destructive mt-2">{searchError}</p>
      )}
    </div>
  );
}
