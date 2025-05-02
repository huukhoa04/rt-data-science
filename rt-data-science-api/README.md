# API Note

## Pinecone Pagination

* [X] API Flow:

1. `{{PINECONE_HOST}}/vectors/list?limit=100&namespace={{namespace}}&paginationToken={{pToken}}` => `response.vectors.map(item => "ids:" + item.ids)`
2. Use mapped list above for querying params:
   + `{{PINECONE_HOST}}/vectors/fetch?ids={{id-1}}&ids={{id-2}}&...&namespace={{namespace}}`
3. Foreach `response.vectors`'s pair, map `metadata`.
