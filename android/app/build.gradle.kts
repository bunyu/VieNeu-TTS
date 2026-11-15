plugins {
    alias(libs.plugins.android.app)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
}
// An Android app that reads EPUB books aloud in Vietnamese with VieNeu-TTS v3 Nano, on the phone.
android {
    namespace = "io.github.bunyu.reader"
    compileSdk = 36
    ndkVersion = "27.1.12297006"
    defaultConfig {
        applicationId = "io.github.bunyu.reader"
        minSdk = 29
        targetSdk = 36
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
        base.archivesName.set("Doc truyen")
        versionCode = 1
        versionName = "0.1.0"
    }
    buildTypes {
        release {
            // Signed with this PC's debug key, the one every earlier build carried, so the release installs
            // over them and keeps the books, voices and the downloaded model. Not for a store.
            signingConfig = signingConfigs.getByName("debug")
            // Left off until a shrunk build has been tried on a phone: ONNX Runtime and the phonemiser are reached through JNI.
            isMinifyEnabled = false
        }
    }
    buildFeatures { compose = true }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    sourceSets["main"].jniLibs.srcDir(layout.buildDirectory.dir("generated/jniLibs"))
    sourceSets["main"].assets.srcDir(layout.buildDirectory.dir("generated/licences"))
    packaging { jniLibs.useLegacyPackaging = false }
}
// The release is for phones only: the x86_64 libraries serve emulators and make up 46 MB of the file.
androidComponents { onVariants(selector().withBuildType("release")) { it.packaging.jniLibs.excludes.add("**/x86_64/**") } }
kotlin { jvmToolchain(21); compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

// The Vietnamese phonemiser is Rust (native/g2p). It needs rustup with the Android targets and cargo-ndk:
//   rustup target add aarch64-linux-android x86_64-linux-android && cargo install cargo-ndk
val g2pSource = layout.projectDirectory.dir("native/g2p")
val buildG2p by tasks.registering(Exec::class) {
    inputs.files(g2pSource.file("Cargo.toml"), g2pSource.file("Cargo.lock"))
    inputs.dir(g2pSource.dir("src"))
    val out = layout.buildDirectory.dir("generated/jniLibs")
    outputs.dir(out)
    workingDir = g2pSource.asFile
    environment("ANDROID_NDK_HOME", android.ndkDirectory.absolutePath)
    val cargo = if (org.gradle.internal.os.OperatingSystem.current().isWindows) "cargo.exe" else "cargo"
    commandLine(cargo, "ndk", "--platform", "29", "-t", "arm64-v8a", "-t", "x86_64", "-o", out.get().asFile.absolutePath, "build", "--release", "--locked")
    // sea-g2p is linked into libreader_g2p; cargo-ndk also copies the crate's own C library, which nothing loads.
    doLast { delete(fileTree(out) { include("**/libsea_g2p_rs.so") }) }
}
// The app shows its own licence and the credits for the model and phonemiser.
val bundleLicences by tasks.registering(Copy::class) {
    from(rootDir.resolve("../LICENSE")) { into("licences"); rename { "LICENSE.txt" } }
    into(layout.buildDirectory.dir("generated/licences"))
}
tasks.named("preBuild") { dependsOn(bundleLicences, buildG2p) }
// SeaG2pTest loads a host build of the bridge (`cargo build --release` in native/g2p) when its folder is given.
tasks.withType<Test>().configureEach {
    providers.gradleProperty("reader.g2p.host").orNull?.let { systemProperty("java.library.path", rootDir.resolve(it).absolutePath) }
    System.getenv("READER_G2P_DICT")?.let { environment("READER_G2P_DICT", it) }
    System.getenv("READER_NANO_DIR")?.let { environment("READER_NANO_DIR", it) }
}

dependencies {
    implementation(libs.activity)
    implementation(libs.lifecycle)
    implementation(libs.core)
    implementation(libs.coroutines)
    implementation(libs.serialization)
    implementation(libs.media)
    implementation(libs.media.session)
    implementation(libs.onnxruntime.android)
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.foundation)
    implementation(libs.compose.material)
    implementation(libs.compose.icons)
    testImplementation(libs.junit)
    testImplementation(libs.coroutines.test)
    // The same Java API as the Android runtime, so the engine port is tested off the phone.
    testImplementation(libs.onnxruntime.jvm)
}
