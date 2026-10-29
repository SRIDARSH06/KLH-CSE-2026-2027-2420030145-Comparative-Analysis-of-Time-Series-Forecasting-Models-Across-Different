import { request as httpRequest } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { Socket } from "node:net";
import net from "node:net";

const STREAMLIT_HOST = "127.0.0.1";
const STREAMLIT_PORT = 5000;

function proxyHeaders(headers: IncomingMessage["headers"]): Record<string, string | string[]> {
  const output: Record<string, string | string[]> = {};
  for (const [name, value] of Object.entries(headers)) {
    if (value !== undefined && name.toLowerCase() !== "host") {
      output[name] = value;
    }
  }
  output.host = `${STREAMLIT_HOST}:${STREAMLIT_PORT}`;
  return output;
}

export function proxyHttpRequest(req: IncomingMessage, res: ServerResponse): void {
  const upstream = httpRequest(
    {
      hostname: STREAMLIT_HOST,
      port: STREAMLIT_PORT,
      method: req.method,
      path: req.url,
      headers: proxyHeaders(req.headers),
    },
    (upstreamResponse) => {
      const headers = { ...upstreamResponse.headers };
      res.writeHead(upstreamResponse.statusCode ?? 502, headers);
      upstreamResponse.pipe(res);
    },
  );

  upstream.on("error", () => {
    if (!res.headersSent) {
      res.statusCode = 502;
      res.end("The Streamlit dashboard is not ready yet.");
    } else {
      res.end();
    }
  });
  req.pipe(upstream);
}

function formatUpgradeHeaders(req: IncomingMessage): string {
  const lines = [`GET ${req.url ?? "/"} HTTP/1.1`];
  for (const [name, value] of Object.entries(proxyHeaders(req.headers))) {
    const values = Array.isArray(value) ? value : [value];
    for (const item of values) {
      lines.push(`${name}: ${item}`);
    }
  }
  return `${lines.join("\r\n")}\r\n\r\n`;
}

export function proxyWebSocket(
  req: IncomingMessage,
  clientSocket: Socket,
  head: Buffer,
): void {
  const upstreamSocket = net.connect(STREAMLIT_PORT, STREAMLIT_HOST, () => {
    upstreamSocket.write(formatUpgradeHeaders(req));
    if (head.length > 0) {
      upstreamSocket.write(head);
    }
    clientSocket.pipe(upstreamSocket).pipe(clientSocket);
  });

  upstreamSocket.on("error", () => {
    clientSocket.destroy();
  });
  clientSocket.on("error", () => {
    upstreamSocket.destroy();
  });
}