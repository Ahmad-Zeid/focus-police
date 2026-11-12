plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "edu.aub.focuspolice.watch"
    compileSdk = 35

    defaultConfig {
        applicationId = "edu.aub.focuspolice.watch"
        minSdk = 30          // Wear OS 3 (Galaxy Watch5 Pro shipped with Wear OS 3.5)
        targetSdk = 34       // Wear OS 5; also runs on Wear OS 4 and 6
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        // A debug build is signed with the local debug key: no account, installs over adb.
        release { isMinifyEnabled = false }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
}
