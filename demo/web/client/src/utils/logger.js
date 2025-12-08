// Logger utilities for formatted logging

const pad2 = (value) => value.toString().padStart(2, '0');
const pad3 = (value) => value.toString().padStart(3, '0');

/**
 * Format timestamp as YYYY-MM-DD HH:MM:SS.mmm
 */
export const formatLocalTimestamp = () => {
  const d = new Date();
  const year = d.getFullYear();
  const month = pad2(d.getMonth() + 1);
  const day = pad2(d.getDate());
  const hours = pad2(d.getHours());
  const minutes = pad2(d.getMinutes());
  const seconds = pad2(d.getSeconds());
  const millis = pad3(d.getMilliseconds());
  return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}.${millis}`;
};

/**
 * Parse timestamp string to Date object
 */
export const parseTimestamp = (value) => {
  if (!value) {
    return new Date();
  }
  if (/\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}/.test(value)) {
    return new Date(value.replace(' ', 'T'));
  }
  return new Date(value);
};

/**
 * Create a logger instance that manages log entries
 */
export class Logger {
  constructor() {
    this.entries = [];
    this.sequence = 0;
    this.listeners = [];
  }

  /**
   * Add a listener for log updates
   */
  addListener(callback) {
    this.listeners.push(callback);
  }

  /**
   * Remove a listener
   */
  removeListener(callback) {
    this.listeners = this.listeners.filter(cb => cb !== callback);
  }

  /**
   * Notify all listeners of log updates
   */
  notifyListeners() {
    this.listeners.forEach(callback => callback(this.entries));
  }

  /**
   * Add a log entry
   */
  log(message, timestamp) {
    const finalTimestamp = timestamp || formatLocalTimestamp();
    const entry = {
      timestamp: finalTimestamp,
      date: parseTimestamp(finalTimestamp),
      message,
      seq: this.sequence += 1,
    };
    
    this.entries.push(entry);
    this.entries.sort((a, b) => {
      const diff = a.date.getTime() - b.date.getTime();
      return diff !== 0 ? diff : a.seq - b.seq;
    });

    // Keep only last 400 entries
    if (this.entries.length > 400) {
      this.entries.splice(0, this.entries.length - 400);
    }

    this.notifyListeners();
  }

  /**
   * Clear all log entries
   */
  clear() {
    this.entries = [];
    this.notifyListeners();
  }

  /**
   * Get formatted log string
   */
  getFormattedLogs() {
    return this.entries
      .map(item => `[${item.timestamp}] ${item.message}`)
      .join('\n');
  }
}
