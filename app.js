

const express = require('express');
const { exec } = require('child_process');
const path = require('path');

const app = express();

// Replace with your IP address (server computer's IP on the network)
const SERVER_IP = 'localhost';
const PORT = 3000;

// Route to trigger PyQt5 login UI
app.get('/login-ui', (req, res) => {
  const scriptPath = path.join(__dirname, 'loginform.py');

  exec(`python "${scriptPath}"`, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error launching PyQt5: ${error.message}`);
      res.status(500).send(" Failed to open PyQt5 UI.");
      return;
    }
    if (stderr) {
      console.error(`stderr: ${stderr}`);
    }

    console.log(`stdout: ${stdout}`);
    res.send(" PyQt5 login UI launched on the server.");
  });
});

// Bind to 0.0.0.0 to allow connections from other computers

app.listen(PORT, '0.0.0.0', () => {
  console.log(` Server running at http://${SERVER_IP}:${PORT}`);
  console.log(` Visit http://${SERVER_IP}:${PORT}/login-ui to open PyQt5 UI`);
});
