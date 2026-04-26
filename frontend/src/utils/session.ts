/**
 * SessionManager — Handles persistent, time-expiring session tokens.
 */

const SESSION_KEY = 'recsys_session_token';
const EXPIRATION_TIME_MS = 30 * 60 * 1000; // 30 minutes

interface SessionToken {
  sessionId: string;
  lastInteraction: number;
}

export const SessionManager = {
  /**
   * Retrieves the current sessionId or creates a new one if expired/absent.
   */
  getOrCreateSessionId(): string {
    const raw = localStorage.getItem(SESSION_KEY);
    const now = Date.now();

    if (raw) {
      try {
        const token: SessionToken = JSON.parse(raw);
        if (now - token.lastInteraction < EXPIRATION_TIME_MS) {
          // Valid session, update interaction time
          this.updateInteraction();
          return token.sessionId;
        }
      } catch (e) {
        console.error('Failed to parse session token:', e);
      }
    }

    // New or expired session
    const newSessionId = crypto.randomUUID();
    this.saveSession(newSessionId, now);
    return newSessionId;
  },

  /**
   * Checks if the stored session is currently expired.
   */
  isExpired(): boolean {
    const raw = localStorage.getItem(SESSION_KEY);
    if (!raw) return true;
    try {
      const token: SessionToken = JSON.parse(raw);
      return (Date.now() - token.lastInteraction) >= EXPIRATION_TIME_MS;
    } catch {
      return true;
    }
  },

  /**
   * Updates the last interaction timestamp to keep the session alive.
   */
  updateInteraction() {
    const raw = localStorage.getItem(SESSION_KEY);
    if (raw) {
      try {
        const token: SessionToken = JSON.parse(raw);
        token.lastInteraction = Date.now();
        localStorage.setItem(SESSION_KEY, JSON.stringify(token));
      } catch (e) {
        // Fallback to new session if parsing fails
        this.getOrCreateSessionId();
      }
    }
  },

  /**
   * Forces a session reset.
   */
  resetSession(): string {
    const newId = crypto.randomUUID();
    this.saveSession(newId, Date.now());
    return newId;
  },

  /**
   * Internal helper to save session state.
   */
  saveSession(sessionId: string, timestamp: number) {
    const token: SessionToken = { sessionId, lastInteraction: timestamp };
    localStorage.setItem(SESSION_KEY, JSON.stringify(token));
  }
};
