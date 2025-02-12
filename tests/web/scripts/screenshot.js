const { Builder, By, until } = require('selenium-webdriver');
const fs = require('fs');
const path = require('path');

// Function to take a screenshot with Selenium WebDriver
async function takeScreenshot(browserName, resolutionWidth, resolutionHeight) {
    const screenshotDir = path.join(__dirname, '../references');
    if (!fs.existsSync(screenshotDir)) {
        fs.mkdirSync(screenshotDir, { recursive: true });
    }

    // Create a new instance of the WebDriver for the specified browser
    let driver = await new Builder().forBrowser(browserName).build();

    try {
        // Set the browser window size to a large size to avoid any clipping or resizing issues
        await driver.manage().window().setRect({ width: 1800, height: 1000 });

        // Navigate to your local website
        await driver.get('http://localhost:8000/gloss_webpage');  // Update with your local URL

        // Wait for the canvas to be present in the DOM
        let canvas = await driver.wait(until.elementLocated(By.css('canvas')), 10000);

        // Log the initial size of the canvas
        const initialSize = await driver.executeScript(`
            const canvas = document.querySelector('canvas');
            return canvas ? { width: canvas.clientWidth, height: canvas.clientHeight } : null;
        `);
        console.log(`Initial canvas size: ${initialSize.width}x${initialSize.height}`);

        // Force the canvas size via JavaScript to ensure it matches the desired dimensions
        await driver.executeScript(`
            const canvas = document.querySelector('canvas');
            if (canvas) {
                console.log('Setting canvas dimensions');
                canvas.style.width = '${resolutionWidth}px';
                canvas.style.height = '${resolutionHeight}px';
                canvas.width = ${resolutionWidth};
                canvas.height = ${resolutionHeight};
            } else {
                console.error('Canvas not found');
            }
        `);

        // Wait a moment to allow the resize to take effect
        await driver.sleep(10000);

        // Verify and log the actual canvas size after setting it
        const canvasSize = await driver.executeScript(`
            const canvas = document.querySelector('canvas');
            return canvas ? { width: canvas.clientWidth, height: canvas.clientHeight, drawingWidth: canvas.width, drawingHeight: canvas.height } : null;
        `);
        console.log(`Actual canvas size (style): ${canvasSize.width}x${canvasSize.height}`);
        console.log(`Actual canvas drawing size: ${canvasSize.drawingWidth}x${canvasSize.drawingHeight}`);

        if (canvasSize.width !== resolutionWidth || canvasSize.height !== resolutionHeight) {
            console.error(`Error: Canvas size does not match the desired resolution of ${resolutionWidth}x${resolutionHeight}`);
            throw new Error(`Canvas size mismatch: expected ${resolutionWidth}x${resolutionHeight}, but got ${canvasSize.width}x${canvasSize.height}`);
        }

        // Capture the screenshot of the canvas
        const screenshotPath = path.join(screenshotDir, `reference_${resolutionWidth}x${resolutionHeight}.png`);
        let screenshot = await canvas.takeScreenshot();
        fs.writeFileSync(screenshotPath, screenshot, 'base64');
        console.log(`Screenshot saved at ${screenshotPath}`);

    } catch (error) {
        console.error('Error:', error);
    } finally {
        await driver.quit();
    }
}

// List of resolutions at 2:1 aspect ratio
const resolutions = [
    { width: 512, height: 256 },
    { width: 1024, height: 512 },
    { width: 1400, height: 700 },
];

// Generate screenshots for different resolutions
(async () => {
    for (const resolution of resolutions) {
        console.log(`Processing resolution: ${resolution.width}x${resolution.height}`);
        await takeScreenshot('chrome', resolution.width, resolution.height);
    }
})();
