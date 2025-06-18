import hashlib
import json
import os
import time
from datetime import datetime, date
from web3 import Web3

# --- Wallet setup ---
pKey   = '0x58cc6dda0b164c045b02b75ad8f3cb472645b9d863a83129a1f6ca560fd141c7'
wal_02 = '0x3a07279c76DbDb90833241AcFC7f32C54ee281eD'

# Connect to Sepolia via Infura
sepolia_rpc = "https://sepolia.infura.io/v3/fde74b6244e74502b31d2062bdc69d58"
w3 = Web3(Web3.HTTPProvider(sepolia_rpc))

if not w3.is_connected():
    raise Exception("❌ Failed to connect to Sepolia")

sender = w3.eth.account.from_key(pKey)

def hash_data(obj):
    def default_serializer(x):
        if isinstance(x, (datetime, date)):
            return x.isoformat()
        elif hasattr(x, 'tolist'):
            return x.tolist()
        elif isinstance(x, bytes):
            return x.decode("utf-8", errors="ignore")
        else:
            return str(x)

    if isinstance(obj, (dict, list)):
        serialized = json.dumps(obj, sort_keys=True, default=default_serializer).encode("utf-8")
    elif isinstance(obj, str):
        serialized = obj.encode("utf-8")
    elif hasattr(obj, 'to_json'):
        serialized = obj.to_json().encode("utf-8")
    elif isinstance(obj, bytes):
        serialized = obj
    else:
        serialized = json.dumps(obj, default=default_serializer).encode("utf-8")

    return hashlib.sha256(serialized).hexdigest()

def send_hash_to_sepolia(hash_hex, nonce):
    data_bytes = w3.to_bytes(hexstr=hash_hex)
    tx = {
        "nonce": nonce,
        "to": wal_02,
        "value": w3.to_wei(0, "ether"),
        "gas": 50000,
        "gasPrice": w3.eth.gas_price,
        "data": data_bytes,
        "chainId": 11155111
    }
    signed_tx = w3.eth.account.sign_transaction(tx, pKey)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    return w3.to_hex(tx_hash)

def log_checkpoints(checkpoints: dict, save_dir="artifacts"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    protocol = {"timestamp": timestamp, "protocol": {}}
    base_nonce = w3.eth.get_transaction_count(sender.address)

    for i, (key, value) in enumerate(checkpoints.items()):
        try:
            hash_hex = hash_data(value)
            nonce = base_nonce + i
            tx_hash = send_hash_to_sepolia(hash_hex, nonce)
            protocol["protocol"][key] = tx_hash
            print(f"✅ {key} sent: {tx_hash}")
            time.sleep(2)
        except Exception as e:
            print(f"❌ Failed at {key}: {e}")

    log_path = os.path.join(save_dir, f"checkpoint_log_{timestamp}.json")
    with open(log_path, "w") as f:
        json.dump(protocol, f, indent=2)
    print(f"📦 Protocol log saved to {log_path}")
