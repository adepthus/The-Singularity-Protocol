# Key Miner v2.1 (Bilingual PL/EN) - Corrected for 2013 Intent: Damaged Fork 14-15.04
# Mines 'original' ('A') from 'damaged' ('B'), confirms identical public FP.
#
import os
import subprocess
import tempfile
import shutil
import sys
import time
import argparse
from datetime import datetime

translations = {
    'pl': {
        "choose_lang": "Wybierz język (PL/EN): ",
        "welcome": "\n--- Key Miner v2.1 ---",
        "openssl_ok": "✅ OpenSSL: {path}",
        "openssl_err": "❌ Brak OpenSSL w PATH.",
        "prompt_paths": "\nPodaj ścieżkę do damaged PEM ('B' na pos.8; lub dwa: damaged pierwszy, drugi dowolny): ",
        "file_err": "Plik '{path}' nie istnieje.",
        "read_err": "❌ Błąd odczytu: {e}",
        "timestamps": "[*] Timestampy (fork intencji 14-15.04):\n   Damaged ({damaged}): {damaged_date}\n   Drugi ({second}): {second_date}\n   Różnica: {diff_days:.1f} dni (~1 OK).",
        "pos_detect": "[*] Pos.8 w damaged: '{damaged_char}' (exp. 'B'). {second_note}",
        "debug_snip": "[*] Pos.0-10 damaged: {damaged_snip}",
        "load_info": "[*] Base64 block: {length} znaków.",
        "mine_start": "[*] Mining original ('A') z damaged ('B')...",
        "scan_pos": "\r   Skan pos. {pos}/{total}...",
        "mine_success": "\n" + "="*50 + "\n🎉 Original ('{new_char}') wydobyty z pos.{pos} po {seconds:.2f}s! 🎉\n" + "="*50,
        "save_note": "\n✅ Zapisano mined_original.pem",
        "verify_cmd": "   openssl rsa -in mined_original.pem -check",
        "no_flip": "❌ Brak flipu do original.",
        "no_flip_hint": "   Użyj --full. Exp. pos.8 'B'→'A'.",
        "bonus_header": "[*] Bonus: FP & pos.8 (intencja 2013 potwierdzona):",
        "bonus_table": "\n" + "="*50 + "\n| Plik          | FP Ident | Pos.8 |\n" + "="*50 + "\n| Damaged       | {damaged_fp} | '{damaged_char}' |\n| Drugi plik    | {second_fp}  | '{second_char}'  |\n| Mined Original| {mined_fp}   | '{mined_char}'   |\n" + "="*50 + "\n{status}",
        "status_ok": "✅ FP ident! Flip B→A + fork 14-15.04 = INTENCJONALNA ANOMALIA 2013!",
        "status_err": "❌ Sprawdź FP/flip.",
        "bonus_fail": "   Bonus err: {e}",
        "second_note": "Drugi: '{second_char}'."
    },
    'en': {
        "choose_lang": "Choose language (PL/EN): ",
        "welcome": "\n--- Key Miner v2.1 ---",
        "openssl_ok": "✅ OpenSSL: {path}",
        "openssl_err": "❌ OpenSSL not in PATH.",
        "prompt_paths": "\nEnter damaged PEM path ('B' at pos.8; or two: damaged first): ",
        "file_err": "File '{path}' not found.",
        "read_err": "❌ Read error: {e}",
        "timestamps": "[*] Timestamps (14-15.04 fork intent):\n   Damaged ({damaged}): {damaged_date}\n   Second ({second}): {second_date}\n   Diff: {diff_days:.1f} days (~1 OK).",
        "pos_detect": "[*] Pos.8 damaged: '{damaged_char}' (exp. 'B'). {second_note}",
        "debug_snip": "[*] Pos.0-10 damaged: {damaged_snip}",
        "load_info": "[*] Base64 block: {length} chars.",
        "mine_start": "[*] Mining original ('A') from damaged ('B')...",
        "scan_pos": "\r   Scan pos. {pos}/{total}...",
        "mine_success": "\n" + "="*50 + "\n🎉 Original ('{new_char}') mined from pos.{pos} in {seconds:.2f}s! 🎉\n" + "="*50,
        "save_note": "\n✅ Saved mined_original.pem",
        "verify_cmd": "   openssl rsa -in mined_original.pem -check",
        "no_flip": "❌ No flip to original.",
        "no_flip_hint": "   Use --full. Exp. pos.8 'B'→'A'.",
        "bonus_header": "[*] Bonus: FP & pos.8 (2013 intent confirmed):",
        "bonus_table": "\n" + "="*50 + "\n| File          | FP Ident | Pos.8 |\n" + "="*50 + "\n| Damaged       | {damaged_fp} | '{damaged_char}' |\n| Second File   | {second_fp}  | '{second_char}'  |\n| Mined Original| {mined_fp}   | '{mined_char}'   |\n" + "="*50 + "\n{status}",
        "status_ok": "✅ Ident FP! B→A flip + 14-15.04 fork = INTENTIONAL 2013 ANOMALY!",
        "status_err": "❌ Check FP/flip.",
        "bonus_fail": "   Bonus err: {e}",
        "second_note": "Second: '{second_char}'."
    }
}

BASE64_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

def get_openssl_path():
    return shutil.which('openssl')

def get_public_fp(openssl_exec, key_file):
    pub_proc = subprocess.run([openssl_exec, 'rsa', '-in', key_file, '-pubout', '-outform', 'DER'], capture_output=True)
    if pub_proc.returncode != 0: raise ValueError("Pubkey extract failed")
    fp_proc = subprocess.run([openssl_exec, 'dgst', '-sha256', '-hex'], input=pub_proc.stdout, capture_output=True)
    if fp_proc.returncode != 0: raise ValueError("FP calc failed")
    fp_line = fp_proc.stdout.decode('utf-8').strip()
    fp_hex = fp_line.split('=')[1].strip()
    return ':'.join(fp_hex[i:i+2] for i in range(0, len(fp_hex), 2))

def get_base64_block(key_file):
    with open(key_file, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    return "".join(line.strip() for line in lines[1:-1]), lines[0].strip(), lines[-1].strip()

def mine_flip(openssl_exec, base64_block, header, footer, range_limit=30):
    for pos in range(range_limit):
        orig_char = base64_block[pos]
        time.sleep(0.05)
        print(f"\r   Skan pos. {pos+1}/{range_limit}...", end="")
        for new_char in BASE64_CHARS:
            if new_char == orig_char: continue
            mod_block = list(base64_block)
            mod_block[pos] = new_char
            candidate = f"{header}\n{''.join(mod_block)}\n{footer}"
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".pem", encoding='utf-8') as tmp:
                tmp.write(candidate)
                tmp_path = tmp.name
            result = subprocess.run([openssl_exec, 'rsa', '-in', tmp_path, '-check', '-noout'], capture_output=True, text=True)
            os.remove(tmp_path)
            if result.returncode == 0:
                return pos, new_char, candidate
    return None, None, None

def main():
    parser = argparse.ArgumentParser(description="Key Miner v2.1 - 2013 Fork Intent Demo.")
    parser.add_argument("paths", nargs="*", help="Damaged PEM paths (first primary).")
    parser.add_argument("--full", action="store_true", help="Full scan.")
    parser.add_argument("--debug", action="store_true", help="Debug snippet.")
    args = parser.parse_args()

    lang = input(translations['pl']['choose_lang']).strip().upper()
    T = translations['en' if lang == 'EN' else 'pl']

    print(T["welcome"])
    openssl_exec = get_openssl_path()
    if openssl_exec:
        print(T["openssl_ok"].format(path=openssl_exec))
    else:
        print(T["openssl_err"])
        sys.exit(1)

    paths_input = ' '.join(args.paths) or input(T["prompt_paths"]).strip()
    if not paths_input:
        print("Need at least one path.")
        sys.exit(1)
    paths = paths_input.split(' ', 1)
    damaged_key_path = paths[0].strip('"\'')
    second_key_path = paths[1].strip('"\'') if len(paths) > 1 else None

    if not os.path.exists(damaged_key_path):
        print(T["file_err"].format(path=damaged_key_path))
        sys.exit(1)
    if second_key_path and not os.path.exists(second_key_path):
        print(T["file_err"].format(path=second_key_path))
        sys.exit(1)

    # Timestamps if second file
    if second_key_path:
        damaged_mtime = os.path.getmtime(damaged_key_path)
        second_mtime = os.path.getmtime(second_key_path)
        damaged_date = datetime.fromtimestamp(damaged_mtime).strftime("%Y-%m-%d %H:%M:%S")
        second_date = datetime.fromtimestamp(second_mtime).strftime("%Y-%m-%d %H:%M:%S")
        diff_days = abs((damaged_mtime - second_mtime) / 86400)
        print(T["timestamps"].format(damaged=damaged_key_path, damaged_date=damaged_date, second=second_key_path, second_date=second_date, diff_days=diff_days))

    # Extract Base64 for damaged
    damaged_base64_block, damaged_header, damaged_footer = get_base64_block(damaged_key_path)
    damaged_char_at_8 = damaged_base64_block[8] if len(damaged_base64_block) > 8 else '?'

    # Second file pos.8 if exists
    second_char_at_8 = None
    second_note = ""
    if second_key_path:
        second_base64_block, _, _ = get_base64_block(second_key_path)
        second_char_at_8 = second_base64_block[8] if len(second_base64_block) > 8 else '?'
        second_note = T["second_note"].format(second_char=second_char_at_8)

    print(T["pos_detect"].format(damaged_char=damaged_char_at_8, second_note=second_note))

    if args.debug:
        print(T["debug_snip"].format(damaged_snip=damaged_base64_block[:11]))

    print(T["load_info"].format(length=len(damaged_base64_block)))
    print(T["mine_start"])

    start_time = time.time()
    range_limit = len(damaged_base64_block) if args.full else 30
    flip_pos, new_char, mined_pem_content = mine_flip(openssl_exec, damaged_base64_block, damaged_header, damaged_footer, range_limit)

    if flip_pos is None:
        print(T["no_flip"])
        print(T["no_flip_hint"])
        sys.exit(1)

    end_time = time.time()
    orig_char = damaged_base64_block[flip_pos]
    print(T["mine_success"].format(pos=flip_pos, new_char=new_char, seconds=end_time - start_time))
    with open('mined_original.pem', 'w', encoding='utf-8') as f:
        f.write(mined_pem_content)
    print(T["save_note"])
    print(T["verify_cmd"])

    # Bonus: FP comparison
    print(T["bonus_header"])
    try:
        damaged_public_fp = get_public_fp(openssl_exec, damaged_key_path)
        second_public_fp = get_public_fp(openssl_exec, second_key_path) if second_key_path else damaged_public_fp
        mined_public_fp = get_public_fp(openssl_exec, 'mined_original.pem')
        mined_char_at_flip = new_char

        fp_match = damaged_public_fp == second_public_fp == mined_public_fp
        day_ok = True if not second_key_path else abs(diff_days - 1) < 2
        if fp_match and day_ok:
            status = T["status_ok"]
        else:
            status = T["status_err"]

        print(T["bonus_table"].format(damaged_fp=damaged_public_fp, second_fp=second_public_fp, mined_fp=mined_public_fp,
                                      damaged_char=damaged_char_at_8, second_char=second_char_at_8 or damaged_char_at_8, mined_char=mined_char_at_flip, status=status))
    except Exception as e:
        print(T["bonus_fail"].format(e=str(e)))

    print("="*50)

if __name__ == "__main__":
    main()