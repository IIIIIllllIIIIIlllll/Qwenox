running_service_processes() {
  local proc exe name pid args arg match found=1
  local -a argv
  for proc in /proc/[0-9]*; do
    exe="$(readlink -- "$proc/exe" 2>/dev/null)" || continue
    name="${exe##*/}"
    name="${name% (deleted)}"
    match=0
    case "$name" in
      qwenox*|flash_serve) match=1 ;;
      python*|pypy*)
        argv=()
        while IFS= read -r -d '' arg; do argv+=("$arg"); done <"$proc/cmdline" 2>/dev/null
        for arg in "${argv[@]}"; do
          if [[ "${arg##*/}" == serve_api.py ]]; then
            match=1
            break
          fi
        done
        ;;
    esac
    [[ "$match" == 1 ]] || continue
    pid="${proc##*/}"
    args="$(tr '\0' ' ' <"$proc/cmdline" 2>/dev/null)" || continue
    printf '%s %s\n' "$pid" "$args"
    found=0
  done
  return "$found"
}
