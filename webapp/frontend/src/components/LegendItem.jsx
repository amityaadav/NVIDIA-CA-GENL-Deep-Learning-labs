import { useEffect, useRef, useState } from "react";
import { registerPopover, closeOtherPopovers } from "../lib/popovers.js";

/**
 * A legend entry (colored dot + label) with a clickable "i" that opens a
 * detailed explanation popover. Closes on click-outside, a second click, or
 * when any other "i" popover in the app is opened.
 */
export default function LegendItem({ dotClass, label, title, children, className = "", ariaLabel }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const closeRef = useRef(() => setOpen(false)); // stable closer for the registry

  useEffect(() => registerPopover(closeRef.current), []);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const toggle = () => {
    setOpen((v) => {
      if (!v) closeOtherPopovers(closeRef.current); // opening → close the rest
      return !v;
    });
  };

  return (
    <span className="legend" ref={ref}>
      {dotClass && <i className={`dot ${dotClass}`} />}
      <span className={`legend-text ${className}`}>
        {label}
        <button
          className={`legend-info ${open ? "on" : ""}`}
          onClick={toggle}
          aria-expanded={open}
          aria-label={ariaLabel || (typeof label === "string" ? `What is ${label}?` : "More information")}
        >
          i
        </button>
      </span>
      {open && (
        <div className="legend-pop">
          <span className="legend-pop-title">{title}</span>
          <div className="legend-pop-body">{children}</div>
        </div>
      )}
    </span>
  );
}
