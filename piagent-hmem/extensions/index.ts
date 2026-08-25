/**
 * piagent-hmem — HMEM Hybrid Memory Extension for Pi Agent
 *
 * Connects Pi Agent to the HMEM Server (c/s architecture) to provide
 * long-term memory with semantic search, knowledge graph, and reflection.
 *
 * Configuration (in priority order):
 *   1. Environment variables: PIAGENT_HMEM_API_URL, PIAGENT_HMEM_API_KEY, PIAGENT_HMEM_NAMESPACE
 *   2. .piagent-hmem.json in the project root
 *   3. /hmem config command (runtime only)
 *
 * Default namespace: "piagent-default"
 */

import type {
	ExtensionAPI,
	ExtensionContext,
	ToolExecutionContext,
	ToolExecutionSignal,
} from "@earendil-works/pi-coding-agent";
import { Type, type TSchema } from "typebox";
import { HmemClient } from "./client.ts";
import * as Tools from "./tools.ts";

// ── Constants ────────────────────────────────────────────────────

const EXTENSION_ID = "piagent-hmem";
const CONFIG_FILE_NAME = ".piagent-hmem.json";
const DEFAULT_NAMESPACE = "piagent-default";

// ── Config loading ───────────────────────────────────────────────

interface PersistedConfig {
	apiUrl: string;
	apiKey: string;
	namespace: string;
}

function loadConfigFromEnv(): Partial<PersistedConfig> {
	const cfg: Partial<PersistedConfig> = {};
	try {
		const apiUrl = (process as any).env.PIAGENT_HMEM_API_URL?.trim();
		const apiKey = (process as any).env.PIAGENT_HMEM_API_KEY?.trim();
		const namespace = (process as any).env.PIAGENT_HMEM_NAMESPACE?.trim();
		if (apiUrl) cfg.apiUrl = apiUrl;
		if (apiKey) cfg.apiKey = apiKey;
		if (namespace) cfg.namespace = namespace;
	} catch {
		/* ignore in non-Node envs */
	}
	return cfg;
}

function loadConfigFromFile(cwd: string): PersistedConfig | undefined {
	try {
		const fs = require("node:fs") as typeof import("fs");
		const path = require("node:path") as typeof import("path");
		const filePath = path.join(cwd, CONFIG_FILE_NAME);
		if (!fs.existsSync(filePath)) return undefined;
		return JSON.parse(fs.readFileSync(filePath, "utf-8")) as PersistedConfig;
	} catch {
		return undefined;
	}
}

function saveConfigToFile(cwd: string, cfg: PersistedConfig): void {
	try {
		const fs = require("node:fs") as typeof import("fs");
		const path = require("node:path") as typeof import("path");
		const filePath = path.join(cwd, CONFIG_FILE_NAME);
		fs.writeFileSync(filePath, JSON.stringify(cfg, null, 2));
	} catch {
		// Non-fatal — continue without saving
	}
}

// ── Tool registration helper ─────────────────────────────────────

interface ToolDef {
	description: string;
	parameters: TSchema;
	execute: (
		toolCallId: string,
		params: Record<string, unknown>,
		signal: ToolExecutionSignal,
		onUpdate: (update: unknown) => void,
		ctx: ToolExecutionContext,
	) => Promise<{
		content: Array<{ type: "text"; text: string }>;
		details: Record<string, unknown>;
	}>;
}

function registerTool(pi: ExtensionAPI, name: string, tool: ToolDef): void {
	pi.registerTool({
		name,
		label: name,
		description: tool.description,
		parameters: tool.parameters,
		execute: tool.execute,
	});
}

// ── Extension factory ────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
	// Session-scoped state
	let client: HmemClient | undefined;
	let initialized = false;

	function getClient(cwd: string): HmemClient {
		if (!client) {
			const envCfg = loadConfigFromEnv();
			const fileCfg = cwd ? loadConfigFromFile(cwd) : undefined;
			const merged: Partial<PersistedConfig> = {
				apiUrl: envCfg.apiUrl ?? fileCfg?.apiUrl ?? "http://localhost:8000",
				apiKey: envCfg.apiKey ?? fileCfg?.apiKey ?? "",
				namespace: envCfg.namespace ?? fileCfg?.namespace ?? DEFAULT_NAMESPACE,
			};
			client = new HmemClient(merged);
		}
		return client;
	}

	// ── Lifecycle ──────────────────────────────────────────────

	pi.on("session_start", async (_event: unknown, ctx: ExtensionContext) => {
		const c = getClient(ctx.cwd);
		initialized = false;
		const health = await c.health();
		if (!health.ok) {
			ctx.ui.notify(
				`⚠️ HMEM server not reachable at ${c.getConfig().apiUrl}`,
				"warning",
			);
			initialized = false;
		} else {
			initialized = true;
			const stats = await c.stats(c.getConfig().namespace);
			const total = stats.ok ? (stats.data?.total_memories ?? "?") : "?";
			ctx.ui.notify(
				`✅ HMEM connected — namespace: ${c.getConfig().namespace}, ${total} memories`,
				"info",
			);
		}
		ctx.ui.setStatus(
			EXTENSION_ID,
			initialized ? "🧠 HMEM ready" : "⚠️ HMEM unreachable",
		);
	});

	pi.on("session_shutdown", async () => {
		initialized = false;
		client = undefined;
	});

	// ── /hmem command ──────────────────────────────────────────

	pi.registerCommand("hmem", {
		description:
			"HMEM memory management — /hmem, /hmem help, /hmem config, /hmem search <q>, /hmem stats, /hmem reflect",
		handler: async (args: string, ctx: ExtensionContext) => {
			const c = getClient(ctx.cwd);
			const parts = (args ?? "").trim().split(/\s+/);
			const cmd = parts[0]?.toLowerCase() || "help";
			const rest = parts.slice(1).join(" ");

			switch (cmd) {
				case "help": {
					ctx.ui.notify(
						[
							"HMEM commands:",
							"  /hmem              — show this help",
							"  /hmem config       — show current config",
							"  /hmem config set <key> <value> — set config (apiUrl/apiKey/namespace)",
							"  /hmem stats        — memory statistics",
							"  /hmem reflect      — trigger reflection",
							"  /hmem search <q>   — quick search (limit 5)",
							"  /hmem list [type]  — list memories",
							"  /hmem models       — list mental models",
							"  /hmem namespaces   — list all namespaces",
						].join("\n"),
						"info",
					);
					return;
				}

				case "config": {
					if (parts.length >= 3 && parts[1] === "set") {
						const key = parts[2];
						const value = parts.slice(3).join(" ");
						if (value) {
							// SAFETY: config keys are arbitrary user-supplied strings (apiUrl/apiKey/namespace),
							// so the dynamic key cannot be typed against the Partial<Record<string, string>> shape.
							c.updateConfig({ [key]: value } as unknown as Partial<
								Record<string, string>
							>);
							const saved = {
								apiUrl: c.getConfig().apiUrl,
								apiKey: c.getConfig().apiKey ? "****" : "",
								namespace: c.getConfig().namespace,
							};
							saveConfigToFile(ctx.cwd, saved);
							ctx.ui.notify(`✅ Config set: ${key}=${value}`, "info");
						} else {
							ctx.ui.notify("❌ /hmem config set <key> <value>", "error");
						}
						return;
					}
					const cfg = c.getConfig();
					ctx.ui.notify(
						[
							"HMEM Config:",
							`  apiUrl:    ${cfg.apiUrl}`,
							`  apiKey:    ${cfg.apiKey ? "****" : "(empty)"}`,
							`  namespace: ${cfg.namespace}`,
						].join("\n"),
						"info",
					);
					return;
				}

				case "stats": {
					const result = await Tools.handleStats(c, {});
					ctx.ui.notify(result, "info");
					return;
				}

				case "reflect": {
					const result = await Tools.handleReflect(c, {});
					ctx.ui.notify(result, "info");
					return;
				}

				case "search": {
					if (!rest) {
						ctx.ui.notify("❌ /hmem search <query>", "error");
						return;
					}
					const result = await Tools.handleSearch(c, { query: rest, limit: 5 });
					ctx.ui.notify(result, "info");
					return;
				}

				case "list": {
					const memoryType = parts[1];
					const result = await Tools.handleList(
						c,
						memoryType ? { memory_type: memoryType } : {},
					);
					ctx.ui.notify(result, "info");
					return;
				}

				case "models":
				case "mental-models": {
					const result = await Tools.handleMentalModels(c, {});
					ctx.ui.notify(result, "info");
					return;
				}

				case "namespaces": {
					const result = await Tools.handleNamespaces(c);
					ctx.ui.notify(result, "info");
					return;
				}

				default:
					ctx.ui.notify(`Unknown subcommand: ${cmd}. Try /hmem help`, "warning");
			}
		},
	});

	// ── System prompt context ──────────────────────────────────

	pi.on("before_agent_start", async (event: any) => {
		if (!initialized) return { systemPrompt: event.systemPrompt };

		const c = getClient(event.cwd ?? "");
		const stats = await c.stats(c.getConfig().namespace);
		if (!stats.ok) return { systemPrompt: event.systemPrompt };

		const s = stats.data!;
		const contextBlock = [
			`# HMEM Memory (namespace: ${s.namespace})`,
			`Connected to HMEM server (${c.getConfig().apiUrl}). ${s.total_memories} memories. Embeddings: ${s.embedding_enabled ? "ON" : "OFF"}.`,
			`By type: ${Object.entries(s.by_type ?? {})
				.map(([k, v]) => `${k}:${v}`)
				.join(", ")}`,
			`Use tools: hmem_write, hmem_search, hmem_list, hmem_get, hmem_delete, hmem_stats, hmem_reflect, hmem_models, hmem_graph, hmem_namespaces.`,
			`Memory hierarchy: observation > experience → reflect → insight → mental_model. Write only observation/experience. Insights & models are auto-generated by hmem_reflect.`,
		].join("\n");

		return { systemPrompt: event.systemPrompt + "\n\n" + contextBlock };
	});

	// ── Prefetch: inject relevant memories into conversation ────

	pi.on("context", async (event: any) => {
		if (!initialized || !client) return;

		// Look for the user's most recent message text
		const messages = event.messages as Array<{ role: string; content: any }>;
		let userText = "";
		for (let i = messages.length - 1; i >= 0; i--) {
			const m = messages[i];
			if (m.role === "user" && typeof m.content === "string") {
				userText = m.content;
				break;
			}
		}

		if (!userText.trim() || userText.length > 500) return;

		const context = await Tools.prefetchMemories(client, userText, 3);
		if (!context) return;

		return {
			messages: [{ role: "system" as const, content: context }, ...messages],
		};
	});

	// ── Register tools ───────────────────────────────────────────

	registerTool(pi, "hmem_write", {
		description:
			"Store a memory in HMEM. Provide content, and optionally memory_type (observation or experience). NOTE: insight/mental_model are read-only — auto-generated by reflection engine.",
		parameters: Type.Object({
			content: Type.String({ description: "The fact/experience to remember" }),
			memory_type: Type.Optional(
				Type.String({
					description:
						"Type: observation (default) or experience. insight/mental_model not allowed — they're auto-generated by hmem_reflect.",
				}),
			),
			mem_action: Type.Optional(
				Type.String({
					description: "Action type (e.g. code_generation, debug)",
				}),
			),
			mem_context: Type.Optional(
				Type.Record(Type.String(), Type.Any(), {
					description: "JSON context metadata",
				}),
			),
			mem_outcome: Type.Optional(
				Type.Record(Type.String(), Type.Any(), {
					description: "JSON outcome metadata",
				}),
			),
			mem_metadata: Type.Optional(
				Type.Record(Type.String(), Type.Any(), {
					description: "Arbitrary JSON metadata",
				}),
			),
			namespace: Type.Optional(
				Type.String({ description: "Override default namespace" }),
			),
		}),
		async execute(_id, params, _signal, _update, _ctx) {
			const c = getClient(_ctx.cwd);
			const result = await Tools.handleWrite(c, params as Record<string, unknown>);
			return {
				content: [{ type: "text" as const, text: result }],
				details: {},
			};
		},
	});

	registerTool(pi, "hmem_search", {
		description:
			"Hybrid semantic search across HMEM memories. Uses keyword + vector + rerank. Returns top-K relevant memories.",
		parameters: Type.Object({
			query: Type.String({ description: "Natural language search query" }),
			limit: Type.Optional(
				Type.Integer({ description: "Max results (default 10, max 50)" }),
			),
			use_rerank: Type.Optional(
				Type.Boolean({
					description: "Use cross-encoder rerank (default true)",
				}),
			),
			min_score: Type.Optional(
				Type.Number({
					description:
						"Minimum relevance score threshold (default 0.1, set 0 to disable filtering)",
				}),
			),
			namespace: Type.Optional(
				Type.String({ description: "Override default namespace" }),
			),
		}),
		async execute(_id, params, _signal, _update, _ctx) {
			const c = getClient(_ctx.cwd);
			const result = await Tools.handleSearch(
				c,
				params as Record<string, unknown>,
			);
			return {
				content: [{ type: "text" as const, text: result }],
				details: {},
			};
		},
	});

	registerTool(pi, "hmem_list", {
		description:
			"List recently stored memories, optionally filtered by memory_type.",
		parameters: Type.Object({
			limit: Type.Optional(
				Type.Integer({ description: "Max results (default 20)" }),
			),
			offset: Type.Optional(Type.Integer({ description: "Pagination offset" })),
			memory_type: Type.Optional(
				Type.String({
					description: "Filter: observation, experience, insight, mental_model",
				}),
			),
			namespace: Type.Optional(
				Type.String({ description: "Override default namespace" }),
			),
		}),
		async execute(_id, params, _signal, _update, _ctx) {
			const c = getClient(_ctx.cwd);
			const result = await Tools.handleList(c, params as Record<string, unknown>);
			return {
				content: [{ type: "text" as const, text: result }],
				details: {},
			};
		},
	});

	registerTool(pi, "hmem_get", {
		description: "Get a single memory by ID.",
		parameters: Type.Object({
			memory_id: Type.Integer({ description: "Memory ID to retrieve" }),
			namespace: Type.Optional(
				Type.String({ description: "Override default namespace" }),
			),
		}),
		async execute(_id, params, _signal, _update, _ctx) {
			const c = getClient(_ctx.cwd);
			const result = await Tools.handleGet(c, params as Record<string, unknown>);
			return {
				content: [{ type: "text" as const, text: result }],
				details: {},
			};
		},
	});

	registerTool(pi, "hmem_delete", {
		description: "Delete a memory by ID.",
		parameters: Type.Object({
			memory_id: Type.Integer({ description: "Memory ID to delete" }),
			namespace: Type.Optional(
				Type.String({ description: "Override default namespace" }),
			),
		}),
		async execute(_id, params, _signal, _update, _ctx) {
			const c = getClient(_ctx.cwd);
			const result = await Tools.handleDelete(
				c,
				params as Record<string, unknown>,
			);
			return {
				content: [{ type: "text" as const, text: result }],
				details: {},
			};
		},
	});

	registerTool(pi, "hmem_stats", {
		description:
			"Get HMEM memory statistics (total count, by type, embedding status).",
		parameters: Type.Object({
			namespace: Type.Optional(
				Type.String({ description: "Override default namespace" }),
			),
		}),
		async execute(_id, params, _signal, _update, _ctx) {
			const c = getClient(_ctx.cwd);
			const result = await Tools.handleStats(c, params as Record<string, unknown>);
			return {
				content: [{ type: "text" as const, text: result }],
				details: {},
			};
		},
	});

	registerTool(pi, "hmem_reflect", {
		description:
			"Trigger HMEM reflection engine to generate insights and mental models from stored experiences.",
		parameters: Type.Object({
			namespace: Type.Optional(
				Type.String({ description: "Override default namespace" }),
			),
		}),
		async execute(_id, params, _signal, _update, _ctx) {
			const c = getClient(_ctx.cwd);
			const result = await Tools.handleReflect(
				c,
				params as Record<string, unknown>,
			);
			return {
				content: [{ type: "text" as const, text: result }],
				details: {},
			};
		},
	});

	registerTool(pi, "hmem_models", {
		description:
			"List mental models and insights from reflection. These capture learned patterns about your preferences and workflows.",
		parameters: Type.Object({
			limit: Type.Optional(
				Type.Integer({ description: "Max results (default 20)" }),
			),
			namespace: Type.Optional(
				Type.String({ description: "Override default namespace" }),
			),
		}),
		async execute(_id, params, _signal, _update, _ctx) {
			const c = getClient(_ctx.cwd);
			const result = await Tools.handleMentalModels(
				c,
				params as Record<string, unknown>,
			);
			return {
				content: [{ type: "text" as const, text: result }],
				details: {},
			};
		},
	});

	registerTool(pi, "hmem_graph", {
		description:
			"Get the knowledge graph (nodes + edges) for the current namespace.",
		parameters: Type.Object({
			namespace: Type.Optional(
				Type.String({ description: "Override default namespace" }),
			),
		}),
		async execute(_id, params, _signal, _update, _ctx) {
			const c = getClient(_ctx.cwd);
			const result = await Tools.handleGraph(c, params as Record<string, unknown>);
			return {
				content: [{ type: "text" as const, text: result }],
				details: {},
			};
		},
	});

	registerTool(pi, "hmem_namespaces", {
		description: "List all available HMEM namespaces and their memory counts.",
		parameters: Type.Object({}),
		async execute(_id, _params, _signal, _update, _ctx) {
			const c = getClient(_ctx.cwd);
			const result = await Tools.handleNamespaces(c);
			return {
				content: [{ type: "text" as const, text: result }],
				details: {},
			};
		},
	});
}
