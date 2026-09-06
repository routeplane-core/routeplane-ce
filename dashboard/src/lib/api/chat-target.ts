/** Explicit single-provider selection only; endpoint configuration stays server-side. */
export function validChatTarget(model: string, provider: string): boolean {
  return model.trim().length > 0 && /^[a-zA-Z0-9_-]+$/.test(provider.trim());
}
