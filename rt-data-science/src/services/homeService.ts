import { environment } from "../environments/environment";
import { apiEndpoints } from "../constants/apiEndpoints";

export const getMovieList = async (token?: string) => {
  const url = `${environment.apiUrl}${apiEndpoints.movies.list(token)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Lỗi ${res.status}: ${res.statusText}`);
  return await res.json();
};

export const getRecommendedMovies = async (query: string) => {
  const url = `${environment.apiUrl}${apiEndpoints.movies.recommend}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: query || "recommended movies", top_k: 5 }),
  });
  if (!res.ok) throw new Error(`Lỗi ${res.status}: ${res.statusText}`);
  return await res.json();
};
