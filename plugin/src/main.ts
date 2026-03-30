import { Plugin } from 'obsidian';

export default class LocusPlugin extends Plugin {
    async onload() {
        console.log('Locus plugin loaded');
    }

    onunload() {
        console.log('Locus plugin unloaded');
    }
}
