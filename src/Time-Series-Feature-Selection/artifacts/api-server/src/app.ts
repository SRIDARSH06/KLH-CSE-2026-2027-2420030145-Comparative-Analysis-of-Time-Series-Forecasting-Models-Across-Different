import express, { type Express } from "express";
import cors from "cors";
import pinoHttp from "pino-http";
import router from "./routes";
import { logger } from "./lib/logger";
import { proxyHttpRequest } from "./preview-proxy";

const app: Express = express();

app.use(
  pinoHttp({
    logger,
    serializers: {
      req(req) {
        return {
          id: req.id,
          method: req.method,
          url: req.url?.split("?")[0],
        };
      },
      res(res) {
        return {
          statusCode: res.statusCode,
        };
      },
    },
  }),
);
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use("/api", router);
// The workspace preview has no root web artifact. Forward root traffic to the
// Streamlit workflow while keeping the API service available under /api.
app.use((req, res) => {
  if (req.url?.startsWith("/api")) {
    res.status(404).json({ message: "API route not found" });
    return;
  }
  proxyHttpRequest(req, res);
});

export default app;
