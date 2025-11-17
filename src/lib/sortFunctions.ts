import type { GenericEntry } from "@/types";

// Sort by date
export const sortByDate = (entries: GenericEntry[]): GenericEntry[] => {
  const sortedEntries = entries.sort(
    (a: any, b: any) => {
      // 优先使用 publishedAt，然后是 date，最后是 createdAt
      const dateA = a.data.publishedAt || a.data.date || a.data.createdAt;
      const dateB = b.data.publishedAt || b.data.date || b.data.createdAt;
      return new Date(dateB).valueOf() - new Date(dateA).valueOf();
    }
  );
  return sortedEntries;
};

//sort by update
export const sortByUpdate = (entries: GenericEntry[]): GenericEntry[] => {
  const sortedEntries = entries.sort(
    (a: any, b: any) =>
      new Date(b.data.updatedAt && b.data.updatedAt).valueOf() -
      new Date(a.data.updatedAt && a.data.updatedAt).valueOf(),
  );
  return sortedEntries;
};

// Sort by title
export const sortByTitle = (entries: GenericEntry[]): GenericEntry[] => {
  const sortedEntries = entries.sort((a: any, b: any) =>
    a.data.title.localeCompare(b.data.title),
  );
  return sortedEntries;
};

// Sort by random
export const sortByRandom = (entries: GenericEntry[]): GenericEntry[] => {
  const sortedEntries = entries.sort(() => Math.random() - 0.5);
  return sortedEntries;
};
