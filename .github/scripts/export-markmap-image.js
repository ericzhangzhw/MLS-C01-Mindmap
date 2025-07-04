// .github/scripts/export-markmap-image.js
const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
  // Get file paths from command line arguments
  const htmlFilePath = process.argv[2];
  const outputPngPath = process.argv[3];

  if (!htmlFilePath || !outputPngPath) {
    console.error('Usage: node export-markmap-image.js <html_file_path> <output_png_path>');
    process.exit(1);
  }

  const absoluteHtmlPath = path.resolve(htmlFilePath);

  // Launch a headless browser
  // The '--no-sandbox' flag is required to run in a container like GitHub Actions
  const browser = await puppeteer.launch({ args: ['--no-sandbox', '--disable-setuid-sandbox'] });
  const page = await browser.newPage();

  // Open the local HTML file
  await page.goto(`file://${absoluteHtmlPath}`, { waitUntil: 'networkidle0' });

  // Find the <svg> element which contains the Markmap visualization
  const svgElement = await page.waitForSelector('svg');

  // Take a screenshot of just the SVG element
  if (svgElement) {
    await svgElement.screenshot({
      path: outputPngPath,
      omitBackground: true, // Make the background transparent
    });
    console.log(`Screenshot saved to ${outputPngPath}`);
  } else {
    console.error('Could not find the SVG element to screenshot.');
    process.exit(1);
  }

  // Close the browser
  await browser.close();
})();