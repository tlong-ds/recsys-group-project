function normalizeOrigin(origin: string): string {
  return origin.endsWith('/') ? origin.slice(0, -1) : origin
}

function stripApiPrefix(pathname: string): string {
  const stripped = pathname.replace(/^\/api/, '')
  return stripped.length > 0 ? stripped : '/'
}

export const onRequest = async (context: { request: Request; env: { API_ORIGIN?: string } }) => {
  const { request, env } = context
  const sourceUrl = new URL(request.url)
  if (!env.API_ORIGIN) {
    return new Response('API_ORIGIN is not configured for Cloudflare Pages Functions.', { status: 500 })
  }

  const upstreamOrigin = normalizeOrigin(env.API_ORIGIN)
  const upstreamPath = stripApiPrefix(sourceUrl.pathname)
  const upstreamUrl = new URL(`${upstreamPath}${sourceUrl.search}`, upstreamOrigin)

  const headers = new Headers(request.headers)
  headers.delete('host')
  const init: RequestInit = {
    method: request.method,
    headers,
    redirect: 'follow',
  }
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    init.body = request.body
  }

  try {
    const response = await fetch(upstreamUrl, init)

    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown upstream fetch error'
    return new Response(
      `API proxy upstream request failed. Check API_ORIGIN and backend health. Target: ${upstreamOrigin}. Error: ${message}`,
      { status: 502 },
    )
  }
}
