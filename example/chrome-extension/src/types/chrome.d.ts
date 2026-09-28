/**
 * The subset of the Chrome extensions API this extension uses, declared by
 * hand so the code type-checks without `@types/chrome` installed. With
 * `npm install --save-dev @types/chrome` this file can be deleted and
 * `"types": ["chrome"]` restored in tsconfig.json; the shapes below follow
 * the official documentation (MV3, Chrome 116+).
 */
declare namespace chrome {
  namespace runtime {
    interface MessageSender {
      tab?: tabs.Tab;
      id?: string;
      url?: string;
    }
    function sendMessage(message: unknown): Promise<any>;
    function getURL(path: string): string;
    function openOptionsPage(): Promise<void>;
    const onMessage: {
      addListener(callback: (message: any, sender: MessageSender, sendResponse: (response?: unknown) => void) => boolean | void): void;
    };
    const onInstalled: { addListener(callback: () => void): void };
  }
  namespace storage {
    interface StorageArea {
      get(keys?: string | string[] | null): Promise<Record<string, any>>;
      set(items: Record<string, unknown>): Promise<void>;
      remove(keys: string | string[]): Promise<void>;
    }
    const sync: StorageArea;
    const local: StorageArea;
    const session: StorageArea;
    const onChanged: {
      addListener(callback: (changes: Record<string, { oldValue?: any; newValue?: any }>, area: string) => void): void;
    };
  }
  namespace tabs {
    interface Tab {
      id?: number;
      url?: string;
      title?: string;
      active: boolean;
      windowId: number;
    }
    function query(info: { active?: boolean; currentWindow?: boolean; url?: string | string[] }): Promise<Tab[]>;
    function sendMessage(tabId: number, message: unknown): Promise<any>;
    const onRemoved: { addListener(callback: (tabId: number) => void): void };
  }
  namespace sidePanel {
    function setPanelBehavior(behavior: { openPanelOnActionClick: boolean }): Promise<void>;
    function setOptions(options: { tabId?: number; path?: string; enabled?: boolean }): Promise<void>;
  }
  namespace tabCapture {
    function getMediaStreamId(options: { targetTabId?: number; consumerTabId?: number }): Promise<string>;
  }
  namespace offscreen {
    enum Reason {
      USER_MEDIA = "USER_MEDIA",
      DISPLAY_MEDIA = "DISPLAY_MEDIA",
      DOM_PARSER = "DOM_PARSER",
    }
    function createDocument(parameters: { url: string; reasons: Reason[]; justification: string }): Promise<void>;
    function closeDocument(): Promise<void>;
  }
  namespace permissions {
    function request(permissions: { origins?: string[]; permissions?: string[] }): Promise<boolean>;
  }
}
