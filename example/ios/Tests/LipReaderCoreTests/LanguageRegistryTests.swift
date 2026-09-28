import XCTest
@testable import LipReaderCore

/// Mirrors test_languages_are_reported_honestly in test_lipreader.py, for
/// both states of the bundle: LipNet present, LipNet absent.
final class LanguageRegistryTests: XCTestCase {
    let without = LanguageRegistry(lipNetURL: nil)
    let with = LanguageRegistry(lipNetURL: URL(fileURLWithPath: "/tmp/LipNetGRID.mlpackage"))

    func testLanguagesAreReportedHonestly() {
        let table = Dictionary(uniqueKeysWithValues: without.languages().map { ($0.code, $0) })
        XCTAssertFalse(table["he"]!.visualAvailable)
        XCTAssertTrue(table["he"]!.visualNote.contains("no public visual speech model"))
        XCTAssertNil(table["he"]!.visualModel)
        for code in ["ar", "de", "it", "es", "fr"] {
            let status = table[code]!
            XCTAssertFalse(status.visualAvailable, code)
            XCTAssertNotNil(status.visualModel, code) // a model is named...
            XCTAssertNotNil(status.visualLicense, code) // ...with its licence...
            XCTAssertTrue(status.visualNote.contains("not runnable here"), code) // ...and why it does not run
        }
        XCTAssertFalse(table["en"]!.visualAvailable, "no bundle, no LipNet")
        XCTAssertTrue(table["en"]!.visualNote.contains("lipnet_to_coreml.py"))
        XCTAssertEqual(without.availableLanguages(), [])
    }

    func testResolveThrowsWithReasons() {
        XCTAssertThrowsError(try without.resolve("he")) { error in
            XCTAssertTrue((error as? LanguageUnavailable)?.candidates.isEmpty ?? false)
            XCTAssertTrue(error.localizedDescription.contains("audio modes"))
        }
        XCTAssertThrowsError(try without.resolve("ar")) { error in
            XCTAssertTrue(error.localizedDescription.contains("CC BY-NC"))
        }
        XCTAssertThrowsError(try without.resolve("en")) { error in
            XCTAssertTrue(error.localizedDescription.contains("LipNetGRID"))
        }
    }

    func testBundledLipNetMakesEnglishAvailableAndNothingElse() throws {
        let spec = try with.resolve("en")
        XCTAssertEqual(spec.key, "lipnet-grid")
        XCTAssertEqual(spec.maxFrames, 75)
        XCTAssertEqual(spec.nativeFps, 25)
        XCTAssertEqual(with.availableLanguages(), ["en"])
        XCTAssertTrue(spec.info.vocabulary.contains("51 words"))
        XCTAssertThrowsError(try with.resolve("he"))
        XCTAssertThrowsError(try with.resolve("es"))
    }

    func testAutoLanguageIsAssumedWhenAModelExistsAndUnavailableOtherwise() {
        let assumed = LanguageResolution.resolve(options: Options(mode: .visual, language: "auto"), registry: with, given: nil)
        XCTAssertEqual(assumed.detection, .assumed)
        XCTAssertEqual(assumed.used, "en")
        XCTAssertTrue(assumed.note.contains("no visual language-identification model"))
        let unavailable = LanguageResolution.resolve(options: Options(mode: .visual, language: "he"), registry: with, given: nil)
        XCTAssertEqual(unavailable.detection, .unavailable)
        XCTAssertNil(unavailable.spec)
        XCTAssertTrue(unavailable.warning?.hasPrefix("visual reading disabled") ?? false)
        let audioOnly = LanguageResolution.resolve(options: Options(mode: .audioAttributed, language: "he"), registry: without, given: nil)
        XCTAssertEqual(audioOnly.detection, .requested)
        XCTAssertEqual(audioOnly.used, "he")
        XCTAssertNil(audioOnly.spec, "audio-attributed never loads a visual model")
    }
}
