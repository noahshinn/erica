BINARY = erica-watch

BIN_DIR = /usr/local/bin

all:
	cd ./erica-watch && cargo build --release && mv target/release/$(BINARY) $(BIN_DIR) && cd ../;

uninstall:
	rm $(BIN_DIR)/$(BINARY);
