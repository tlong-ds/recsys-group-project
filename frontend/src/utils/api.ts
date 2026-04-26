/**
 * API client for the RecSys FastAPI backend.
 *
 * Falls back gracefully when the backend is unreachable.
 */

export const API_BASE = import.meta.env.VITE_API_BASE || '/api';
export const API_KEY = import.meta.env.VITE_RECSYS_API_KEY || '';

export interface RecommendRequest {
  session_id?: string;
  item_sequence: number[];
  top_k?: number;
}

export interface RecommendResponse {
  session_id: string | null;
  item_sequence: number[];
  recommendations: number[];
  recommended_products?: ProductInfo[];
}

export interface ProductInfo {
  id: number;
  categoryId: number;
  name: string | null;
  price: number | null;
}

export interface PaginatedProductsResponse {
  items: ProductInfo[];
  total_pages: number;
  current_page: number;
  next_cursor: number | null;
}

/**
 * Enhanced fetch that automatically adds the Authorization header if an API key is present.
 */
export async function fetchWithAuth(
  url: string,
  options: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(options.headers);
  if (API_KEY && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${API_KEY}`);
  }

  return fetch(url, {
    ...options,
    headers,
  });
}

/**
 * Fetch products from the catalog with pagination.
 */
export async function fetchProducts(
  page = 1,
  pageSize = 20,
  categoryId?: number,
): Promise<PaginatedProductsResponse> {
  const params = new URLSearchParams();
  params.append('page', page.toString());
  params.append('page_size', pageSize.toString());
  if (categoryId !== undefined) params.append('category_id', categoryId.toString());

  const res = await fetchWithAuth(`${API_BASE}/products?${params.toString()}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch products: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchRecommendations(
  sequence: number[],
  topK = 10,
  sessionId?: string,
): Promise<RecommendResponse> {
  const res = await fetchWithAuth(`${API_BASE}/recommend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      item_sequence: sequence,
      top_k: topK,
    } satisfies RecommendRequest),
  });

  if (!res.ok) {
    const detail = await res.text().catch(() => 'Unknown error');
    throw new Error(`API ${res.status}: ${detail}`);
  }

  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}
