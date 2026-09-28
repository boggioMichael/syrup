// ShareViewController.swift — "Share to LipReader" from Photos, Files or
// any app that shares a video. The extension cannot run the analysis (it is
// memory- and time-limited), so it copies the video into the App Group
// container's Inbox and the app picks it up when it next becomes active
// (AppModel.pickUpSharedInbox). Nothing else is read or stored.
//
// Setup: both targets need the App Group capability with the same
// identifier; `ShareExtension/Info.plist` limits activation to one movie.
import UIKit
import UniformTypeIdentifiers

enum ShareAppGroup {
    /// PLACEHOLDER: must equal AppGroup.identifier in the app target.
    static let identifier = "group.dev.lipreader.shared"
}

final class ShareViewController: UIViewController {
    private let label = UILabel()

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground
        label.numberOfLines = 0
        label.textAlignment = .center
        label.text = "Saving the video for LipReader…"
        label.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(label)
        NSLayoutConstraint.activate([
            label.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            label.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            label.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 24),
            label.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -24),
        ])
        handleInput()
    }

    private func handleInput() {
        let attachments = (extensionContext?.inputItems as? [NSExtensionItem])?.flatMap { $0.attachments ?? [] } ?? []
        guard let provider = attachments.first(where: { $0.hasItemConformingToTypeIdentifier(UTType.movie.identifier) }) else {
            finish("That item is not a video.", error: true)
            return
        }
        // loadFileRepresentation hands over a temporary file that is gone once the
        // handler returns, so the copy into the inbox happens inside it.
        _ = provider.loadFileRepresentation(forTypeIdentifier: UTType.movie.identifier) { [weak self] url, error in
            guard let self else { return }
            guard let url, error == nil else {
                DispatchQueue.main.async { self.finish("The video could not be read: \(error?.localizedDescription ?? "unknown error")", error: true) }
                return
            }
            do {
                let inbox = try Self.inboxDirectory()
                let destination = inbox.appending(path: "\(UUID().uuidString).\(url.pathExtension.isEmpty ? "mov" : url.pathExtension)")
                try FileManager.default.copyItem(at: url, to: destination)
                DispatchQueue.main.async { self.finish("Saved. Open LipReader to analyse it.", error: false) }
            } catch {
                DispatchQueue.main.async { self.finish("Could not save the video: \(error.localizedDescription)", error: true) }
            }
        }
    }

    private static func inboxDirectory() throws -> URL {
        guard let container = FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: ShareAppGroup.identifier) else {
            throw NSError(domain: "LipReaderShare", code: 1, userInfo: [NSLocalizedDescriptionKey: "the App Group \(ShareAppGroup.identifier) is not configured"])
        }
        let inbox = container.appending(path: "Inbox")
        try FileManager.default.createDirectory(at: inbox, withIntermediateDirectories: true)
        return inbox
    }

    private func finish(_ message: String, error: Bool) {
        label.text = message
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { [weak self] in
            if error {
                self?.extensionContext?.cancelRequest(withError: NSError(domain: "LipReaderShare", code: 2, userInfo: [NSLocalizedDescriptionKey: message]))
            } else {
                self?.extensionContext?.completeRequest(returningItems: nil)
            }
        }
    }
}
