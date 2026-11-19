import fs from "fs/promises";
import os from "os";
import path from "path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { FileScanProgress } from "./streaming_folder_service";
import { streamSlpFiles } from "./streaming_folder_service";

async function collect(folder: string, chunkSize: number): Promise<FileScanProgress[]> {
  const out: FileScanProgress[] = [];
  for await (const progress of streamSlpFiles(folder, chunkSize)) {
    out.push(progress);
  }
  return out;
}

describe("streamSlpFiles", () => {
  let folder: string;

  beforeEach(async () => {
    folder = await fs.mkdtemp(path.join(os.tmpdir(), "stream-slp-"));
  });

  afterEach(async () => {
    await fs.rm(folder, { recursive: true, force: true });
  });

  it("gives each replay file once, in the update after it is found", async () => {
    const replays = Array.from({ length: 23 }, (_, i) => `game${i}.brep`).concat(["UPPER.BREP"]);
    for (const name of [...replays, "notes.txt", "old.slp"]) {
      await fs.writeFile(path.join(folder, name), "");
    }
    await fs.mkdir(path.join(folder, "sub.brep"));

    const updates = await collect(folder, 5);

    // 27 entries at 5 per update: 5 updates on the way, then the last one.
    expect(updates.map((u) => u.scanned)).toEqual([5, 10, 15, 20, 25, 27]);
    expect(updates.map((u) => u.complete)).toEqual([false, false, false, false, false, true]);
    for (const u of updates) {
      expect(u.newSlpFiles.length).toBeLessThanOrEqual(5);
    }
    const all = updates.flatMap((u) => u.newSlpFiles);
    expect(all.sort()).toEqual([...replays].sort());
  });

  it("gives one empty final update for a folder that can't be read", async () => {
    const updates = await collect(path.join(folder, "missing"), 5);
    expect(updates).toEqual([{ scanned: 0, newSlpFiles: [], complete: true }]);
  });
});
