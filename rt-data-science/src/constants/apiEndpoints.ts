export const apiEndpoints = {
  movies: {
    list: (token?: string) =>
      token ? `/movies?pagination_token=${encodeURIComponent(token)}` : "/movies",
    recommend: "/movies/recommend",
  },
};
