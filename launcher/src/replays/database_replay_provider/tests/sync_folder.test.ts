import type { Database } from "@database/schema";
import { initTestDb } from "@database/tests/init_test_db";
import fs from "fs/promises";
import type { Kysely } from "kysely";
import os from "os";
import path from "path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { DatabaseReplayProvider } from "../database_replay_provider";

describe("folder sync", () => {
  let db: Kysely<Database>;
  let provider: DatabaseReplayProvider;
  let folder: string;

  beforeEach(async () => {
    db = await initTestDb();
    provider = new DatabaseReplayProvider(db);
    folder = await fs.mkdtemp(path.join(os.tmpdir(), "sync-folder-"));
  });

  afterEach(async () => {
    await db.destroy();
    await fs.rm(folder, { recursive: true, force: true });
  });

  // The folder scan reports files in batches (200 entries each): a replay found in any batch counts.
  it("keeps every indexed replay still on disk and forgets the one that is gone", async () => {
    const names = Array.from({ length: 450 }, (_, i) => `game${String(i).padStart(3, "0")}.brep`);
    for (const name of names) {
      await fs.writeFile(path.join(folder, name), "");
    }
    await db
      .insertInto("file")
      .values([...names, "deleted.brep"].map((name) => ({ folder, name, size_bytes: 0 })))
      .execute();

    await provider.getAllFilePaths(folder);

    const rows = await db.selectFrom("file").select("name").where("folder", "=", folder).execute();
    expect(rows.map((r) => r.name).sort()).toEqual(names);
  });
});
