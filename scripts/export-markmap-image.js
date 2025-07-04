import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { Transformer } from 'markmap-lib';
import { fillTemplate } from 'markmap-render';
import nodeHtmlToImage from 'node-html-to-image';
import puppeteer from 'puppeteer';

async function renderMarkmap(mdPath, outPath) {
  try {
    const markdown = await readFile(mdPath, 'utf8');

    const transformer = new Transformer();
    const { root, features } = transformer.transform(markdown);
    const assets = transformer.getUsedAssets(features);

    const html =
      fillTemplate(root, assets, {
        jsonOptions: {
          duration: 0,
          maxInitialScale: 5,
        },
      }) +
      `
<style>
  body, #mindmap {
    width: 2400px;
    height: 1800px;
  }
</style>
`;

    const imageBuffer = await nodeHtmlToImage({
      html,
      puppeteer,
      executablePath: process.env.PUPPETEER_EXECUTABLE_PATH, // 👈 passed from GitHub Action
      puppeteerArgs: ['--no-sandbox'],
      type: 'png',
      encoding: 'buffer',
    });

    await writeFile(outPath, imageBuffer);
    console.log(`✅ PNG saved to: ${outPath}`);
  } catch (err) {
    console.error('❌ Error rendering Markmap image:', err);
    process.exit(1);
  }
}

const markdownFile = path.resolve('02-data-analysis/mindmap.md');
const outputImage = path.resolve('02-data-analysis/img/mindmap.png');

renderMarkmap(markdownFile, outputImage);
