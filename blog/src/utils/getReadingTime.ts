/**
 * Estimate how many minutes an article takes to read.
 * CJK characters are counted individually while Latin words are weighted
 * slightly higher to account for spaces and punctuation.
 */
export function getReadingTime(body?: string): number {
  const text = (body ?? "")
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/<[^>]+>/g, " ")
    .trim();

  const cjkCharacters = text.match(/[\u3400-\u9fff]/g)?.length ?? 0;
  const latinWords = text.match(/[A-Za-z0-9]+/g)?.length ?? 0;
  const estimatedUnits = cjkCharacters + latinWords * 2;

  return Math.max(1, Math.ceil(estimatedUnits / 400));
}
