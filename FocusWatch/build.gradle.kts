plugins {
    id("com.android.application") version "8.7.3" apply false
    id("org.jetbrains.kotlin.android") version "2.0.21" apply false
}

// Build output goes outside the project folder, so it never gets committed or synced.
allprojects {
    layout.buildDirectory.set(file("${System.getProperty("user.home")}/Library/Caches/FocusWatch/${project.name}"))
}
