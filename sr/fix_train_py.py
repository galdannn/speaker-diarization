path = "train.py"
content = open(path, encoding="utf-8").read()

old = (
    "    model.change_labels(\n"
    "        new_labels=[str(i) for i in range(num_classes)]\n"
    "    )"
)

new = (
    "    # NeMo 2.7.0: change_labels does not exist — rebuild decoder head manually\n"
    "    import torch.nn as nn\n"
    "    decoder = model.decoder\n"
    "    final_layer, final_name = None, None\n"
    "    for name, module in decoder.named_modules():\n"
    "        if isinstance(module, nn.Linear):\n"
    "            final_layer, final_name = module, name\n"
    "    if final_layer is not None:\n"
    "        in_features = final_layer.in_features\n"
    "        new_linear = nn.Linear(in_features, num_classes)\n"
    "        parts = final_name.split('.')\n"
    "        parent = decoder\n"
    "        for part in parts[:-1]:\n"
    "            parent = getattr(parent, part)\n"
    "        setattr(parent, parts[-1], new_linear.to(next(model.parameters()).device))\n"
    "        log.info(f'Decoder head replaced: {in_features} -> {num_classes} classes')\n"
    "    else:\n"
    "        log.warning('Could not find final Linear layer in decoder')"
)

count = content.count(old)
print(f"Found: {count}")
if count == 1:
    content = content.replace(old, new)
    open(path, "w", encoding="utf-8").write(content)
    print("Done.")
else:
    print("Not found — paste the output to Claude")