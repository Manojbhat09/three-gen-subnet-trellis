module.exports = {
  apps: [
    {
      name: "validation-server",
      script: "validation/serve.py",
      interpreter: "/home/mbhat/miniconda/envs/three-gen-validation/bin/python",
      args: "--host 0.0.0.0 --port 10006",
      cwd: __dirname,
      autorestart: true,
      max_memory_restart: "4G",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
    },
    {
      name: "generation-server",
      script: "flux_hunyuan_sugar_generation_server.py", // Your custom generation script
      interpreter: "/home/mbhat/miniconda/envs/hunyuan3d/bin/python",
      args: "--port 8095", // Assuming it takes a port argument
      cwd: __dirname,
      autorestart: true,
      max_memory_restart: "24G", // Adjust based on your model's VRAM usage
      log_date_format: "YYYY-MM-DD HH:mm:ss",
    },
    {
      name: "miner-neuron",
      script: "neurons/serve_miner.py",
      interpreter: "/home/mbhat/miniconda/envs/three-gen-neurons/bin/python",
      args: [
        "--wallet.name", "test2m3b2",     // <-- EDIT THIS
        "--wallet.hotkey", "t2m3b21",   // <-- EDIT THIS
        "--subtensor.network", "finney",
        "--netuid", "17",
        "--generation.endpoint", "http://127.0.0.1:8095", // Points to your custom generation server
        "--logging.trace"
      ].join(" "),
      cwd: __dirname,
      autorestart: true,
      max_memory_restart: "2G",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      // Delay restart to prevent fast-crashing loops if there's a persistent error
      restart_delay: 10000, 
    }
  ]
}; 