const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

const client = new Client({
    authStrategy: new LocalAuth(),
    puppeteer: {
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-accelerated-2d-canvas',
            '--no-first-run',
            '--no-zygote'
        ]
    }
});

client.on('qr', (qr) => {
    // Generate and scan this code with your phone
    qrcode.generate(qr, { small: true });
    console.log("👆 Please scan the QR code above with your WhatsApp app.");
});

client.on('ready', () => {
    console.log('✅ WhatsApp Bridge is officially LIVE and connected to your number!');
});

client.on('message', async msg => {
    try {
        if (msg.hasMedia) {
            const media = await msg.downloadMedia();
            if (media.mimetype.startsWith('audio/')) {
                console.log(`🎤 Received Voice Note from ${msg.from}`);
                
                // Decode base64 and save to temp file
                const ext = media.mimetype.split('/')[1].split(';')[0];
                const filename = `temp_audio_${Date.now()}.${ext}`;
                fs.writeFileSync(filename, media.data, { encoding: 'base64' });

                const form = new FormData();
                form.append('audio', fs.createReadStream(filename));
                form.append('sender', msg.from);

                // Send to Python FastAPI
                const response = await axios.post('http://127.0.0.1:8000/api/voice', form, {
                    headers: form.getHeaders()
                });

                fs.unlinkSync(filename);
                
                const reply = response.data.final_reply || response.data.reply || response.data.message || "Processed.";
                await msg.reply(reply);
            }
        } 
        else if (msg.body) {
            console.log(`💬 Received Text from ${msg.from}: ${msg.body}`);
            const form = new FormData();
            form.append('query', msg.body);
            
            // For text, we can just use the existing mature endpoint
            const response = await axios.post('http://127.0.0.1:8000/api/web_chat', form, {
                headers: form.getHeaders()
            });

            const reply = response.data.reply || "Processed.";
            await msg.reply(reply);
        }
    } catch (e) {
        console.error('❌ Error handling message:', e.message);
        msg.reply("System Error Processing Request.");
    }
});

client.initialize();
