/**
 * Coordinates the app's info popovers ("i" modals) so only one is open at a
 * time. Each popover registers a close callback; opening one closes the others.
 * Kept framework-free (a plain module-level registry) so popovers rendered in
 * unrelated components — legend keys, the connection note, the weight-image
 * inspector, the canvas layer tooltips — stay mutually exclusive without a
 * shared parent or context.
 */
const closers = new Set();

/** Register a popover's close fn; returns an unregister fn for cleanup. */
export function registerPopover(close) {
  closers.add(close);
  return () => closers.delete(close);
}

/** Close every registered popover except `self` (the one being opened). */
export function closeOtherPopovers(self) {
  for (const close of closers) if (close !== self) close();
}
