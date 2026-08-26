/**
 * HMEM HTTP Client — typed wrapper around the HMEM Server REST API.
 *
 * All API calls return a structured result with `ok: boolean` and
 * either `data` or `error` populated.
 */

export interface HmemConfig {
	apiUrl: string;
	apiKey: string;
	namespace: string;
	/** 分级共享：额外检索的共享库名（如 "shared"），存放用户偏好/心智模型等公共记忆 */
	sharedNs?: string;
}

export interface HmemMemory {
	id: number;
	content: string;
	memory_type: string;
	mem_action?: string;
	mem_context?: Record<string, unknown>;
	mem_outcome?: Record<string, unknown>;
	mem_metadata?: Record<string, unknown>;
	score?: number;
	created_at?: string;
	updated_at?: string;
	parent_id?: number | null;
	hit_count?: number;
	/** 知识库溯源（搜索结果命中知识条目时附带） */
	doc_id?: string;
	doc_title?: string;
	doc_uri?: string;
	chunk_index?: number;
	doc_category?: string;
	doc_tags?: string;
	source?: { doc_id: string; title: string; uri: string; chunk_index: number };
}

export interface HmemSearchResult {
	results: HmemMemory[];
	count: number;
}

export interface HmemStats {
	total_memories: number;
	embedding_enabled: boolean;
	by_type: Record<string, number>;
	namespace: string;
}

export interface HmemReflectResult {
	reflect_count: number;
	status: string;
	stage?: string;
	counts?: Record<string, number>;
}

export interface HmemMentalModel {
	id: number;
	content: string;
	memory_type: "insight" | "mental_model";
	supporting_experiences?: HmemMemory[];
	created_at?: string;
}

export interface HmemGraphData {
	nodes: Array<{ id: number; label: string; type: string }>;
	edges: Array<{ source: number; target: number; relation: string }>;
}

export interface HmemWriteResult {
	memory_id: number;
	content: string;
	namespace: string;
	embedded: boolean;
	auto_reflected: boolean;
}

export interface HmemResult<T> {
	ok: boolean;
	data?: T;
	error?: string;
}

export class HmemClient {
	private config: HmemConfig;

	constructor(config: Partial<HmemConfig> = {}) {
		this.config = {
			apiUrl: config.apiUrl ?? "http://localhost:8000",
			apiKey: config.apiKey ?? "",
			namespace: config.namespace ?? "piagent-default",
			sharedNs: config.sharedNs ?? "",
		};
	}

	/** Update runtime config (e.g. from env vars or /hmem command). */
	updateConfig(partial: Partial<HmemConfig>): void {
		Object.assign(this.config, partial);
	}

	getConfig(): Readonly<HmemConfig> {
		return { ...this.config };
	}

	// ── Internal helpers ──────────────────────────────────────────

	private headers(): Record<string, string> {
		const h: Record<string, string> = { "Content-Type": "application/json" };
		if (this.config.apiKey) h["Authorization"] = `Bearer ${this.config.apiKey}`;
		return h;
	}

	private baseUrl(): string {
		return this.config.apiUrl.replace(/\/+$/, "");
	}

	private async get<T>(
		path: string,
		params?: Record<string, string | number | undefined>,
	): Promise<HmemResult<T>> {
		try {
			const url = new URL(`${this.baseUrl()}${path}`);
			if (params) {
				for (const [k, v] of Object.entries(params)) {
					if (v !== undefined && v !== null && v !== "")
						url.searchParams.set(k, String(v));
				}
			}
			const resp = await fetch(url.toString(), {
				method: "GET",
				headers: this.headers(),
			});
			if (!resp.ok) {
				const text = await resp.text().catch(() => "");
				return {
					ok: false,
					error: `HMEM ${resp.status}: ${text.slice(0, 200)}`,
				};
			}
			const data = (await resp.json()) as T;
			return { ok: true, data };
		} catch (e: unknown) {
			const msg = e instanceof Error ? e.message : String(e);
			return { ok: false, error: `HMEM connection failed: ${msg}` };
		}
	}

	private async post<T>(path: string, body: unknown): Promise<HmemResult<T>> {
		try {
			const resp = await fetch(`${this.baseUrl()}${path}`, {
				method: "POST",
				headers: this.headers(),
				body: JSON.stringify(body),
			});
			if (!resp.ok) {
				const text = await resp.text().catch(() => "");
				return {
					ok: false,
					error: `HMEM ${resp.status}: ${text.slice(0, 200)}`,
				};
			}
			const data = (await resp.json()) as T;
			return { ok: true, data };
		} catch (e: unknown) {
			const msg = e instanceof Error ? e.message : String(e);
			return { ok: false, error: `HMEM connection failed: ${msg}` };
		}
	}

	private async del<T>(
		path: string,
		params?: Record<string, string | number | undefined>,
	): Promise<HmemResult<T>> {
		try {
			const url = new URL(`${this.baseUrl()}${path}`);
			if (params) {
				for (const [k, v] of Object.entries(params)) {
					if (v !== undefined && v !== null && v !== "")
						url.searchParams.set(k, String(v));
				}
			}
			const resp = await fetch(url.toString(), {
				method: "DELETE",
				headers: this.headers(),
			});
			if (!resp.ok) {
				const text = await resp.text().catch(() => "");
				return {
					ok: false,
					error: `HMEM ${resp.status}: ${text.slice(0, 200)}`,
				};
			}
			const data = (await resp.json()) as T;
			return { ok: true, data };
		} catch (e: unknown) {
			const msg = e instanceof Error ? e.message : String(e);
			return { ok: false, error: `HMEM connection failed: ${msg}` };
		}
	}

	// ── Health ─────────────────────────────────────────────────────

	async health(): Promise<HmemResult<{ status: string; version: string }>> {
		return this.get("/health");
	}

	// ── Write ──────────────────────────────────────────────────────

	async writeMemory(opts: {
		content: string;
		memory_type?: string;
		mem_action?: string;
		mem_context?: Record<string, unknown>;
		mem_outcome?: Record<string, unknown>;
		mem_metadata?: Record<string, unknown>;
		namespace?: string;
	}): Promise<HmemResult<HmemWriteResult>> {
		return this.post("/api/v1/memories", {
			content: opts.content,
			namespace: opts.namespace ?? this.config.namespace,
			memory_type: opts.memory_type ?? "observation",
			mem_action: opts.mem_action ?? "",
			mem_context: opts.mem_context ?? {},
			mem_outcome: opts.mem_outcome ?? {},
			mem_metadata: opts.mem_metadata ?? {},
		});
	}

	// ── Search ─────────────────────────────────────────────────────

	async search(opts: {
		query: string;
		limit?: number;
		use_rerank?: boolean;
		min_score?: number;
		namespace?: string;
	}): Promise<HmemResult<HmemSearchResult>> {
		// 分级共享：若配置了 sharedNs 且 ≠ 主库，则一并检索共享库
		const extra: string[] = [];
		const mainNs = opts.namespace ?? this.config.namespace;
		if (this.config.sharedNs && this.config.sharedNs !== mainNs) {
			extra.push(this.config.sharedNs);
		}
		return this.post("/api/v1/search", {
			query: opts.query,
			limit: opts.limit ?? 10,
			namespace: mainNs,
			use_rerank: opts.use_rerank ?? true,
			min_score: opts.min_score,
			extra_namespaces: extra.length ? extra : undefined,
		});
	}

	// ── List ───────────────────────────────────────────────────────

	async listMemories(opts: {
		limit?: number;
		offset?: number;
		memory_type?: string;
		namespace?: string;
	}): Promise<HmemResult<{ results: HmemMemory[]; count: number }>> {
		return this.get("/api/v1/memories", {
			namespace: opts.namespace ?? this.config.namespace,
			limit: opts.limit ?? 20,
			offset: opts.offset ?? 0,
			memory_type: opts.memory_type,
		});
	}

	// ── Get single ─────────────────────────────────────────────────

	async getMemory(
		memoryId: number,
		namespace?: string,
	): Promise<HmemResult<HmemMemory>> {
		return this.get(`/api/v1/memories/${memoryId}`, {
			namespace: namespace ?? this.config.namespace,
		});
	}

	// ── Delete ─────────────────────────────────────────────────────

	async deleteMemory(
		memoryId: number,
		namespace?: string,
	): Promise<HmemResult<{ deleted: boolean; memory_id: number }>> {
		return this.del(`/api/v1/memories/${memoryId}`, {
			namespace: namespace ?? this.config.namespace,
		});
	}

	// ── Stats ──────────────────────────────────────────────────────

	async stats(namespace?: string): Promise<HmemResult<HmemStats>> {
		return this.get("/api/v1/stats", {
			namespace: namespace ?? this.config.namespace,
		});
	}

	// ── Namespaces ─────────────────────────────────────────────────

	async listNamespaces(): Promise<
		HmemResult<{
			namespaces: Array<{ namespace: string; total_memories: number }>;
		}>
	> {
		return this.get("/api/v1/namespaces");
	}

	// ── Reflect ────────────────────────────────────────────────────

	async triggerReflect(
		namespace?: string,
	): Promise<HmemResult<HmemReflectResult>> {
		return this.post("/api/v1/reflect", {
			namespace: namespace ?? this.config.namespace,
		});
	}

	// ── Mental Models ──────────────────────────────────────────────

	async listMentalModels(opts: {
		limit?: number;
		namespace?: string;
	}): Promise<HmemResult<{ results: HmemMentalModel[]; count: number }>> {
		return this.get("/api/v1/mental-models", {
			namespace: opts.namespace ?? this.config.namespace,
			limit: opts.limit ?? 50,
		});
	}

	async getMentalModel(
		modelId: number,
		namespace?: string,
	): Promise<HmemResult<HmemMentalModel & { error?: string }>> {
		return this.get(`/api/v1/mental-models/${modelId}`, {
			namespace: namespace ?? this.config.namespace,
		});
	}

	// ── Graph ──────────────────────────────────────────────────────

	async graph(namespace?: string): Promise<HmemResult<HmemGraphData>> {
		return this.get("/api/v1/graph", {
			namespace: namespace ?? this.config.namespace,
		});
	}

	// ── Logs ───────────────────────────────────────────────────────

	async logs(opts: {
		limit?: number;
		action?: string;
		status?: string;
		namespace?: string;
	}): Promise<
		HmemResult<{ results: Array<Record<string, unknown>>; count: number }>
	> {
		return this.get("/api/v1/logs", {
			namespace: opts.namespace ?? this.config.namespace,
			limit: opts.limit ?? 20,
			action: opts.action,
			status: opts.status,
		});
	}

	// ── Knowledge Base: documents ──────────────────────────────────

	async importDocument(opts: {
		content: string;
		doc_id?: string;
		title?: string;
		uri?: string;
		category?: string;
		tags?: string[];
		chunk_size?: number;
		overlap?: number;
		namespace?: string;
	}): Promise<
		HmemResult<{
			doc_id: string;
			title: string;
			uri: string;
			category: string;
			tags: string[];
			namespace: string;
			chunks: number;
			embedded: number;
			vector_dim: number;
		}>
	> {
		return this.post("/api/v1/documents", {
			content: opts.content,
			namespace: opts.namespace ?? this.config.namespace,
			doc_id: opts.doc_id,
			title: opts.title ?? "",
			uri: opts.uri ?? "",
			category: opts.category ?? "",
			tags: opts.tags ?? [],
			chunk_size: opts.chunk_size,
			overlap: opts.overlap,
		});
	}

	async listDocuments(
		namespace?: string,
	): Promise<
		HmemResult<{
			documents: Array<{
				doc_id: string;
			doc_title: string;
			doc_uri: string;
			category: string;
			tags: string;
			chunk_count: number;
			created_at?: string;
		}>;
			count: number;
			namespace: string;
		}>
	> {
		return this.get("/api/v1/documents", {
			namespace: namespace ?? this.config.namespace,
		});
	}

	async getDocument(
		docId: string,
		namespace?: string,
	): Promise<
		HmemResult<{
			doc_id: string;
			doc_title: string;
			doc_uri: string;
			category: string;
			tags: string;
			chunk_count: number;
			created_at?: string;
			namespace: string;
			chunks: Array<{
				chunk_index: number;
				content: string;
				memory_id: number;
				created_at?: string;
			}>;
		}>
	> {
		return this.get(`/api/v1/documents/${docId}`, {
			namespace: namespace ?? this.config.namespace,
		});
	}

	async deleteDocument(
		docId: string,
		namespace?: string,
	): Promise<
		HmemResult<{ deleted: boolean; doc_id: string; chunks_removed: number }>
	> {
		return this.del(`/api/v1/documents/${docId}`, {
			namespace: namespace ?? this.config.namespace,
		});
	}

	// ── Knowledge Base: single knowledge entries ──────────────────

	async createKnowledge(opts: {
		content: string;
		namespace?: string;
		doc_id?: string;
		title?: string;
		uri?: string;
		category?: string;
		tags?: string[];
		mem_metadata?: Record<string, unknown>;
	}): Promise<
		HmemResult<{ memory_id: number; namespace: string }>
	> {
		return this.post("/api/v1/knowledge", {
			content: opts.content,
			namespace: opts.namespace ?? this.config.namespace,
			doc_id: opts.doc_id,
			title: opts.title ?? "",
			uri: opts.uri ?? "",
			category: opts.category ?? "",
			tags: opts.tags ?? [],
			mem_metadata: opts.mem_metadata ?? {},
		});
	}

	async listKnowledge(opts: {
		namespace?: string;
		limit?: number;
		offset?: number;
		category?: string;
		doc_id?: string;
		tags?: string;
	}): Promise<
		HmemResult<{ namespace: string; count: number; results: HmemMemory[] }>
	> {
		return this.get("/api/v1/knowledge", {
			namespace: opts.namespace ?? this.config.namespace,
			limit: opts.limit ?? 50,
			offset: opts.offset ?? 0,
			category: opts.category,
			doc_id: opts.doc_id,
			tags: opts.tags,
		});
	}

	async listKnowledgeCategories(
		namespace?: string,
	): Promise<
		HmemResult<{
			namespace: string;
			count: number;
			categories: Array<{
				category: string;
				entries: number;
				documents: number;
				tags: string[];
			}>;
		}>
	> {
		return this.get("/api/v1/knowledge/categories", {
			namespace: namespace ?? this.config.namespace,
		});
	}

	// ── Knowledge Base: library-level (knowledge-bases) ───────────

	async createKnowledgeBase(opts: {
		namespace: string;
		title?: string;
		description?: string;
	}): Promise<
		HmemResult<{
			namespace: string;
			title: string;
			description: string;
			db_path: string;
			created: boolean;
		}>
	> {
		return this.post("/api/v1/knowledge-bases", opts);
	}

	async listKnowledgeBases(): Promise<
		HmemResult<{
			knowledge_bases: Array<{
				namespace: string;
				entries: number;
				documents: number;
				categories: number;
				category_list: string[];
			}>;
			count: number;
		}>
	> {
		return this.get("/api/v1/knowledge-bases");
	}

	async deleteKnowledgeBase(
		namespace: string,
		hard = false,
	): Promise<
		HmemResult<{
			deleted: boolean;
			namespace: string;
			hard: boolean;
			db_removed: boolean;
		}>
	> {
		return this.del(`/api/v1/knowledge-bases/${namespace}`, {
			hard: hard ? "true" : undefined,
		});
	}
}
