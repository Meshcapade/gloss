(async () => {
    const { Builder, By, until } = await import('selenium-webdriver');
    const fs = await import('fs');
    const path = await import('path');
    const pixelmatchModule = await import('pixelmatch');
    const pixelmatch = pixelmatchModule.default;
    const { PNG } = await import('pngjs');
    const chrome = await import('selenium-webdriver/chrome.js');
    const firefox = await import('selenium-webdriver/firefox.js');
    const edge = await import('selenium-webdriver/edge.js');
    const safari = await import('selenium-webdriver/safari.js');

    // Global handler for unhandled promise rejections
    // Need this to avoid queue size exceeded error for retries
    process.on('unhandledRejection', (reason, promise) => {
        console.error('Unhandled Rejection. Likely that [BROWSERSTACK_QUEUE_SIZE_EXCEEDED]');
    });

    function determineBestResolution(screenWidth) {
        const resolutions = [1400, 1024, 512];
        return resolutions.reduce((closest, resolution) => {
            return Math.abs(screenWidth - resolution) < Math.abs(screenWidth - closest) ? resolution : closest;
        });
    }

    async function captureScreenshot(driver, desiredWidth, desiredHeight, referencesDir) {
        const currentScreenshotPath = path.join(referencesDir, `screenshot_${desiredWidth}x${desiredHeight}.png`);

        const viewerCanvas = await driver.wait(until.elementLocated(By.css('canvas')), 10000);
        await driver.wait(until.elementIsVisible(viewerCanvas));
        const screenshotData = await viewerCanvas.takeScreenshot();
        fs.writeFileSync(currentScreenshotPath, screenshotData, 'base64');

        return currentScreenshotPath;
    }

    function getBrowserOptions(userAgent) {
        let options;

        if (userAgent.includes('Chrome')) {
            console.log('Detected Browser: Chrome');
            options = new chrome.Options();
            options.addArguments('force-device-scale-factor=1');
            options.addArguments('high-dpi-support=1');
            options.addArguments('window-size=1920,1080');
        } else if (userAgent.includes('Firefox')) {
            console.log('Detected Browser: Firefox');
            options = new firefox.Options();
            options.addArguments('--width=1920');
            options.addArguments('--height=1080');
            options.addArguments('--safe-mode');
            options.setPreference('browser.tabs.remote.autostart', false);
        } else if (userAgent.includes('Edg')) {
            console.log('Detected Browser: Edge');
            options = new edge.Options();
            options.addArguments('force-device-scale-factor=1');
            options.addArguments('high-dpi-support=1');
            options.addArguments('window-size=1920,1080');
        } else if (userAgent.includes('Safari')) {
            console.log('Detected Browser: Safari');
            options = new safari.Options();
        } else {
            console.log('Browser not specifically handled, applying default settings');
            options = null;
        }

        return options;
    }

    async function runTest() {
        const referencesDir = path.join(__dirname, '../references');
        if (!fs.existsSync(referencesDir)) {
            fs.mkdirSync(referencesDir, { recursive: true });
        }

        let driver;

        try {
            driver = await new Builder().build();
            const userAgent = await driver.executeScript('return navigator.userAgent;');
            console.log('Detected user agent:', userAgent);

            const options = getBrowserOptions(userAgent);

            await driver.quit();

            driver = await new Builder()
                .withCapabilities({
                    browserName: userAgent.includes('Chrome') ? 'chrome' :
                        userAgent.includes('Firefox') ? 'firefox' :
                        userAgent.includes('Edg') ? 'MicrosoftEdge' :
                        userAgent.includes('Safari') ? 'safari' :
                        'chrome'
                })
                .setChromeOptions(options)
                .setFirefoxOptions(options)
                .setEdgeOptions(options)
                .setSafariOptions(options)
                .build();

            await driver.get('http://localhost:8000/gloss_webpage');

            if (userAgent.includes('Safari')) {
                await driver.executeScript('window.resizeTo(1920, 1080);');
            }

            const webglEnabled = await driver.executeScript(`
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            return !!gl;
            `);
            console.log('WebGL enabled:', webglEnabled);

            let viewerLoaded = false;
            let maxViewerLoadRetries = 10;

            while (!viewerLoaded && maxViewerLoadRetries > 0) {
                try {
                    viewerLoaded = await driver.executeScript(() => {
                        const canvas = document.querySelector('canvas');
                        return canvas && canvas.clientHeight > 0 && canvas.clientWidth > 0 && canvas.width > 0 && canvas.height > 0;
                    });
                } catch (e) {
                    viewerLoaded = false;
                }

                if (!viewerLoaded) {
                    console.log(`Viewer not loaded yet. Retrying... (${10 - maxViewerLoadRetries + 1})`);
                    await driver.sleep(1000);
                    maxViewerLoadRetries--;
                }
            }

            if (!viewerLoaded) {
                throw new Error('Viewer did not load successfully within the expected time.');
            }

            console.log('Viewer loaded successfully.');
            await driver.sleep(15000);

            const screenWidth = await driver.executeScript('return window.screen.width');
            console.log(`Device screen width: ${screenWidth}px`);

            const resolutions = [1400, 1024, 512];
            let chosenResolutionIndex = 0;

            console.log(`Starting resolution for comparison: ${resolutions[chosenResolutionIndex]}px`);

            let success = false;

            while (!success && chosenResolutionIndex < resolutions.length) {
                const desiredWidth = resolutions[chosenResolutionIndex];
                const desiredHeight = desiredWidth / 2;

                console.log(`Attempting to use resolution: ${desiredWidth}px`);

                const referenceScreenshotPath = path.join(referencesDir, `reference_${desiredWidth}x${desiredHeight}.png`);
                const referenceImage = PNG.sync.read(fs.readFileSync(referenceScreenshotPath));
                console.log(`Reference screenshot dimensions: ${desiredWidth}x${desiredHeight}`);

                if (!/iPhone|iPad/i.test(userAgent)) {
                    try {
                        await driver.manage().window().setRect({ width: desiredWidth + 400, height: desiredHeight + 400 });
                    } catch (e) {
                        console.log('Window resizing is not supported on this device.');
                    }
                }

                await driver.executeScript(`
                    const canvas = document.querySelector('canvas');
                    if (canvas) {
                        canvas.style.width = '${desiredWidth}px';
                        canvas.style.height = '${desiredHeight}px';
                        canvas.width = ${desiredWidth};
                        canvas.height = ${desiredHeight};
                    }
                `);

                await driver.sleep(10000);

                const currentScreenshotPath = await captureScreenshot(driver, desiredWidth, desiredHeight, referencesDir);

                const currentImage = PNG.sync.read(fs.readFileSync(currentScreenshotPath));
                console.log(`Current screenshot dimensions: ${currentImage.width}x${currentImage.height}`);

                if (currentImage.width !== referenceImage.width || currentImage.height !== referenceImage.height) {
                    console.log(`Screenshot dimensions do not match the reference dimensions for resolution: ${desiredWidth}px`);
                    chosenResolutionIndex++;
                } else {
                    const { width, height } = currentImage;
                    const diff = new PNG({ width, height });

                    const diffPixels = pixelmatch(currentImage.data, referenceImage.data, diff.data, width, height, { threshold: 0.2 });

                    const diffPath = path.join(referencesDir, `diff_${desiredWidth}x${desiredHeight}.png`);
                    fs.writeFileSync(diffPath, PNG.sync.write(diff));
                    console.log(`Diff image saved at ${diffPath}`);

                    console.log(`Number of pixels outside threshold: ${diffPixels}`);

                    const threshold = 5000;
                    if (diffPixels > threshold) {
                        console.log('====================================================================================');
                        throw new Error(`[TEST FAILED]: Pixel difference (${diffPixels}) exceeds threshold (${threshold})`);
                        console.log('====================================================================================');
                    } else {
                        console.log('====================================================================');
                        console.log('[TEST PASSED]: Pixel difference is within the acceptable threshold');
                        console.log('====================================================================');
                    }

                    success = true;
                }
            }

            if (!success) {
                throw new Error('Failed to resize canvas to match any desired resolution.');
            }

        } finally {
            if (driver && driver.session_) {
                try {
                    await driver.quit();
                } catch (quitError) {
                    console.log('Custom Error: Failed to quit driver session.');
                }
            }
        }
    }

    async function compareScreenshot(maxRetries = 5) {
        let retryCount = 0;

        while (retryCount < maxRetries) {
            try {
                await runTest();
                break; // Exit loop if the test runs successfully
            } catch (error) {
                if (error.message.includes('BROWSERSTACK_QUEUE_SIZE_EXCEEDED')) {
                    console.log(`Queue size exceeded. Retrying (${retryCount + 1}/${maxRetries})...`);
                    retryCount++;
                    await new Promise(resolve => setTimeout(resolve, 120000)); // Wait for 2 minutes before retrying
                } else {
                    console.log("Custom Error: An error occurred during the screenshot comparison process.");
                    break;
                }
            }
        }

        if (retryCount === maxRetries) {
            console.log('Custom Error: Test failed after maximum retries due to queue size exceeded.');
        }
    }

    compareScreenshot();
})();
