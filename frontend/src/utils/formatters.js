// utils/formatters.js — Number/text formatters

/**
 * Format a number as a percentage string.
 */
export function formatPercent(value, decimals = 1) {
  if (value == null || isNaN(value)) return "—";
  return `${Number(value).toFixed(decimals)}%`;
}

/**
 * Format a large number with commas.
 */
export function formatNumber(value) {
  if (value == null || isNaN(value)) return "—";
  return Number(value).toLocaleString();
}

/**
 * Capitalize the first letter of a string.
 */
export function capitalize(str) {
  if (!str) return "";
  return str.charAt(0).toUpperCase() + str.slice(1);
}
