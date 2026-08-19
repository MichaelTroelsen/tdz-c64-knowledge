"""Memory-map diagram rendering and PDF image extraction.

Split out of wiki_export.py, which was 13,356 lines. These methods are a
mixin on WikiExporter and are unchanged from the originals - they still
reach through `self` for state that lives on the exporter.
"""

from typing import Dict, List
from matplotlib.patches import Rectangle, FancyBboxPatch
from PIL import Image
from pathlib import Path
import fitz  # PyMuPDF for PDF image extraction
import hashlib
import io
# The non-interactive backend and the LaTeX/mathtext settings used to sit at
# the top of wiki_export.py, before anything imported pyplot. They live here
# now because this is the only module that renders a figure, so importing
# this mixin directly still gets them - the old arrangement only worked if
# you came in through wiki_export.
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['text.parse_math'] = False
matplotlib.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt


class DiagramsMixin:
    """Memory-map diagram rendering and PDF image extraction."""

    def _extract_images_from_pdfs(self, title: str, entity: Dict, max_images: int = 6) -> List[Dict]:
        """Extract relevant images from PDF documents for this entity."""
        images = []
        images_dir = self.output_dir / "assets" / "images" / "articles"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Get PDF documents for this entity
        pdf_docs = [doc for doc in entity['documents'][:10] if doc.get('file_path', '').lower().endswith('.pdf')]

        for doc in pdf_docs:
            if len(images) >= max_images:
                break

            try:
                file_path = Path(doc['file_path'])
                if not file_path.exists():
                    continue

                # Open PDF
                pdf_document = fitz.open(str(file_path))

                # Extract images from first 10 pages
                for page_num in range(min(10, len(pdf_document))):
                    if len(images) >= max_images:
                        break

                    page = pdf_document[page_num]
                    image_list = page.get_images(full=True)

                    for img_index, img in enumerate(image_list):
                        if len(images) >= max_images:
                            break

                        try:
                            xref = img[0]
                            base_image = pdf_document.extract_image(xref)
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]

                            # Open with PIL to check size
                            pil_image = Image.open(io.BytesIO(image_bytes))

                            # Skip very small images (likely icons or decorations)
                            if pil_image.width < 100 or pil_image.height < 100:
                                continue

                            # Skip very large images (likely full page scans)
                            if pil_image.width > 2000 or pil_image.height > 2000:
                                # Resize large images
                                pil_image.thumbnail((800, 800), Image.Resampling.LANCZOS)

                            # Generate unique filename
                            img_hash = hashlib.md5(image_bytes).hexdigest()[:12]
                            image_filename = f"{title.lower().replace(' ', '-')}_{img_hash}.{image_ext}"
                            image_path = images_dir / image_filename

                            # Save image
                            if not image_path.exists():
                                if pil_image.width > 800 or pil_image.height > 800:
                                    pil_image.save(str(image_path), quality=85, optimize=True)
                                else:
                                    pil_image.save(str(image_path))

                            images.append({
                                'filename': image_filename,
                                'path': f"../assets/images/articles/{image_filename}",
                                'width': pil_image.width,
                                'height': pil_image.height,
                                'source_doc': doc['title'],
                                'source_page': page_num + 1
                            })

                        except Exception as e:
                            print(f"    Warning: Could not extract image {img_index} from page {page_num}: {e}")
                            continue

                pdf_document.close()

            except Exception as e:
                print(f"    Warning: Could not process PDF {doc.get('title', 'unknown')}: {e}")
                continue

        return images

    def _generate_memory_map_diagrams(self, title: str, category: str) -> List[Dict]:
        """Generate memory map and technical diagrams for C64 components."""
        diagrams = []
        images_dir = self.output_dir / "assets" / "images" / "articles"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Disable LaTeX rendering to allow $ symbols in text
        plt.rcParams['text.usetex'] = False

        title_upper = title.upper()

        # SID Chip Memory Map
        if 'SID' in title_upper:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 14)
            ax.axis('off')

            # Title
            ax.text(5, 13, 'SID Chip Register Map ($D400-$D41F)',
                   ha='center', fontsize=16, fontweight='bold')

            # Voice 1 registers
            y_start = 11
            registers = [
                ('$D400-$D401', 'Voice 1 Frequency', '#4A90E2'),
                ('$D402-$D403', 'Voice 1 Pulse Width', '#4A90E2'),
                ('$D404', 'Voice 1 Control Register', '#4A90E2'),
                ('$D405', 'Voice 1 Attack/Decay', '#4A90E2'),
                ('$D406', 'Voice 1 Sustain/Release', '#4A90E2'),
                # Voice 2
                ('$D407-$D408', 'Voice 2 Frequency', '#50C878'),
                ('$D409-$D40A', 'Voice 2 Pulse Width', '#50C878'),
                ('$D40B', 'Voice 2 Control Register', '#50C878'),
                ('$D40C', 'Voice 2 Attack/Decay', '#50C878'),
                ('$D40D', 'Voice 2 Sustain/Release', '#50C878'),
                # Voice 3
                ('$D40E-$D40F', 'Voice 3 Frequency', '#E76F51'),
                ('$D410-$D411', 'Voice 3 Pulse Width', '#E76F51'),
                ('$D412', 'Voice 3 Control Register', '#E76F51'),
                ('$D413', 'Voice 3 Attack/Decay', '#E76F51'),
                ('$D414', 'Voice 3 Sustain/Release', '#E76F51'),
                # Filter and volume
                ('$D415-$D416', 'Filter Cutoff Frequency', '#9D4EDD'),
                ('$D417', 'Filter Resonance/Routing', '#9D4EDD'),
                ('$D418', 'Filter Mode/Volume', '#9D4EDD'),
            ]

            for i, (addr, desc, color) in enumerate(registers):
                y = y_start - (i * 0.65)
                rect = FancyBboxPatch((0.5, y-0.3), 3.5, 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(2.25, y, addr, ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white')
                ax.text(4.5, y, desc, va='center', fontsize=9)

            filename = 'sid_memory_map.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'SID Register Memory Map',
                'description': 'Complete register layout for the SID sound chip'
            })

        # VIC-II Memory Map
        if 'VIC' in title_upper:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 18)
            ax.axis('off')

            ax.text(5, 17, 'VIC-II Chip Register Map ($D000-$D02E)',
                   ha='center', fontsize=16, fontweight='bold')

            y_start = 15.5
            registers = [
                ('$D000-$D001', 'Sprite 0 X/Y Position', '#E63946'),
                ('$D002-$D003', 'Sprite 1 X/Y Position', '#E63946'),
                ('$D004-$D005', 'Sprite 2 X/Y Position', '#E63946'),
                ('$D006-$D007', 'Sprite 3 X/Y Position', '#E63946'),
                ('$D008-$D009', 'Sprite 4 X/Y Position', '#E63946'),
                ('$D00A-$D00B', 'Sprite 5 X/Y Position', '#E63946'),
                ('$D00C-$D00D', 'Sprite 6 X/Y Position', '#E63946'),
                ('$D00E-$D00F', 'Sprite 7 X/Y Position', '#E63946'),
                ('$D010', 'Sprites 0-7 X MSB', '#E63946'),
                ('$D011', 'Screen Control Register 1', '#4A90E2'),
                ('$D012', 'Raster Counter', '#4A90E2'),
                ('$D015', 'Sprite Enable Register', '#F4A261'),
                ('$D016', 'Screen Control Register 2', '#4A90E2'),
                ('$D017', 'Sprite Y Expansion', '#F4A261'),
                ('$D018', 'Memory Pointers', '#2A9D8F'),
                ('$D019', 'Interrupt Register', '#2A9D8F'),
                ('$D01A', 'Interrupt Enable', '#2A9D8F'),
                ('$D01B', 'Sprite Data Priority', '#F4A261'),
                ('$D01C', 'Sprite Multicolor Mode', '#F4A261'),
                ('$D01D', 'Sprite X Expansion', '#F4A261'),
                ('$D020', 'Border Color', '#9D4EDD'),
                ('$D021', 'Background Color 0', '#9D4EDD'),
                ('$D022-$D023', 'Background Color 1-2', '#9D4EDD'),
                ('$D027-$D02E', 'Sprite 0-7 Colors', '#9D4EDD'),
            ]

            for i, (addr, desc, color) in enumerate(registers):
                y = y_start - (i * 0.65)
                rect = FancyBboxPatch((0.5, y-0.3), 3.5, 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(2.25, y, addr, ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')
                ax.text(4.5, y, desc, va='center', fontsize=8.5)

            filename = 'vic-ii_memory_map.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'VIC-II Register Memory Map',
                'description': 'Complete register layout for the VIC-II graphics chip'
            })

        # Sprite specifications diagram
        if 'SPRITE' in title_upper:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 8)
            ax.axis('off')

            ax.text(5, 7.5, 'C64 Sprite Specifications',
                   ha='center', fontsize=16, fontweight='bold')

            # Draw sprite grid (24x21 pixels)
            sprite_x, sprite_y = 1, 4
            cell_size = 0.15
            for row in range(21):
                for col in range(24):
                    color = '#4A90E2' if (row + col) % 2 == 0 else '#E0E0E0'
                    rect = Rectangle((sprite_x + col*cell_size, sprite_y - row*cell_size),
                                   cell_size, cell_size,
                                   facecolor=color, edgecolor='gray', linewidth=0.3)
                    ax.add_patch(rect)

            ax.text(sprite_x + 1.8, sprite_y + 0.5, '24 pixels wide',
                   ha='center', fontsize=10, fontweight='bold')
            ax.text(sprite_x - 0.5, sprite_y - 1.5, '21\npixels\nhigh',
                   va='center', fontsize=10, fontweight='bold')

            # Specifications table
            specs_x, specs_y = 5.5, 6
            specs = [
                ('Dimensions:', '24 × 21 pixels'),
                ('Total Sprites:', '8 (numbered 0-7)'),
                ('Colors:', '1 color + transparent'),
                ('Multicolor:', '3 colors + transparent'),
                ('X Range:', '0-511 pixels'),
                ('Y Range:', '0-255 pixels'),
                ('Data Size:', '63 bytes per sprite'),
                ('Expansion:', '2x horizontal/vertical'),
            ]

            for i, (label, value) in enumerate(specs):
                y = specs_y - i*0.5
                ax.text(specs_x, y, label, fontsize=10, fontweight='bold')
                ax.text(specs_x + 2, y, value, fontsize=10)

            filename = 'sprite_specs.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'Sprite Technical Specifications',
                'description': 'Visual representation of C64 sprite dimensions and capabilities'
            })

        # CIA chip registers
        if 'CIA' in title_upper:
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 12)
            ax.axis('off')

            ax.text(5, 11.5, 'CIA Chip Register Map',
                   ha='center', fontsize=14, fontweight='bold')
            ax.text(5, 11, 'CIA1: $DC00  |  CIA2: $DD00',
                   ha='center', fontsize=11, style='italic', color='#666666')

            y_start = 10
            registers = [
                ('$00', 'Data Port A', '#4A90E2'),
                ('$01', 'Data Port B', '#4A90E2'),
                ('$02', 'Data Direction Port A', '#50C878'),
                ('$03', 'Data Direction Port B', '#50C878'),
                ('$04-$05', 'Timer A (Low/High)', '#E76F51'),
                ('$06-$07', 'Timer B (Low/High)', '#E76F51'),
                ('$08-$0B', 'Time of Day Clock', '#9D4EDD'),
                ('$0C', 'Serial Shift Register', '#F4A261'),
                ('$0D', 'Interrupt Control', '#2A9D8F'),
                ('$0E', 'Timer A Control', '#E63946'),
                ('$0F', 'Timer B Control', '#E63946'),
            ]

            for i, (addr, desc, color) in enumerate(registers):
                y = y_start - (i * 0.8)
                rect = FancyBboxPatch((1, y-0.35), 2.5, 0.6,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(2.25, y, addr, ha='center', va='center',
                       fontsize=11, fontweight='bold', color='white')
                ax.text(4, y, desc, va='center', fontsize=10)

            filename = 'cia_registers.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'CIA Register Map',
                'description': 'Register layout for the 6526 Complex Interface Adapter'
            })

        # 6502 Processor Status Register
        if '6502' in title_upper or '6510' in title_upper:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_xlim(0, 12)
            ax.set_ylim(0, 8)
            ax.axis('off')

            ax.text(6, 7.5, '6502/6510 Processor Status Register',
                   ha='center', fontsize=16, fontweight='bold')

            # Draw the 8-bit register
            bit_width = 1.2
            start_x = 1.8
            y_pos = 5

            flags = [
                ('7', 'N', 'Negative', '#E63946'),
                ('6', 'V', 'Overflow', '#F4A261'),
                ('5', '-', 'Unused', '#CCCCCC'),
                ('4', 'B', 'Break', '#4A90E2'),
                ('3', 'D', 'Decimal', '#50C878'),
                ('2', 'I', 'Interrupt', '#9D4EDD'),
                ('1', 'Z', 'Zero', '#2A9D8F'),
                ('0', 'C', 'Carry', '#E76F51'),
            ]

            for i, (bit, flag, name, color) in enumerate(flags):
                x = start_x + i * bit_width

                # Draw bit box
                rect = FancyBboxPatch((x, y_pos), bit_width-0.1, 0.8,
                                     boxstyle="round,pad=0.02",
                                     facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(rect)

                # Bit number at top
                ax.text(x + bit_width/2 - 0.05, y_pos + 1.1, f'Bit {bit}',
                       ha='center', fontsize=8, fontweight='bold')

                # Flag letter
                ax.text(x + bit_width/2 - 0.05, y_pos + 0.4, flag,
                       ha='center', va='center', fontsize=20, fontweight='bold', color='white')

                # Flag name below
                ax.text(x + bit_width/2 - 0.05, y_pos - 0.5, name,
                       ha='center', fontsize=9)

            # Add legend
            legend_y = 2.5
            ax.text(6, legend_y + 0.5, 'Status Flags Explained:', ha='center', fontsize=11, fontweight='bold')

            explanations = [
                'N: Set if result is negative (bit 7 = 1)',
                'V: Set on signed overflow',
                'B: Set when BRK instruction executed',
                'D: Decimal mode flag (BCD arithmetic)',
                'I: Interrupt disable flag',
                'Z: Set if result is zero',
                'C: Carry/borrow flag',
            ]

            for i, exp in enumerate(explanations):
                row = i // 2
                col = i % 2
                x = 2 if col == 0 else 7
                y = legend_y - 0.3 - (row * 0.35)
                ax.text(x, y, f'• {exp}', fontsize=8.5, ha='left')

            filename = '6502_status_register.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': '6502 Processor Status Register',
                'description': 'The 8-bit status register showing all processor flags'
            })

        # 1541 Disk Drive Track/Sector Layout
        if '1541' in title_upper:
            fig, ax = plt.subplots(figsize=(11, 8))
            ax.set_xlim(0, 11)
            ax.set_ylim(0, 10)
            ax.axis('off')

            ax.text(5.5, 9.5, '1541 Disk Drive Track/Sector Layout',
                   ha='center', fontsize=16, fontweight='bold')

            # Track zones with different sector counts
            zones = [
                (1, 17, 21, '#4A90E2', 'Tracks 1-17: 21 sectors/track'),
                (18, 24, 19, '#50C878', 'Tracks 18-24: 19 sectors/track'),
                (25, 30, 18, '#F4A261', 'Tracks 25-30: 18 sectors/track'),
                (31, 35, 17, '#E76F51', 'Tracks 31-35: 17 sectors/track'),
            ]

            y_start = 7.5
            for zone_idx, (start_track, end_track, sectors, color, label) in enumerate(zones):
                y = y_start - (zone_idx * 1.5)

                # Zone header
                rect = FancyBboxPatch((1, y), 9, 0.8,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(rect)
                ax.text(5.5, y + 0.4, label, ha='center', va='center',
                       fontsize=11, fontweight='bold', color='white')

                # Show track range
                ax.text(1.5, y - 0.35, f'{end_track - start_track + 1} tracks × {sectors} sectors = {(end_track - start_track + 1) * sectors} sectors',
                       fontsize=9, style='italic')

            # Summary box
            summary_y = 1.5
            total_sectors = (17*21) + (7*19) + (6*18) + (5*17)

            summary_rect = FancyBboxPatch((1.5, summary_y - 1), 8, 1.2,
                                         boxstyle="round,pad=0.1",
                                         facecolor='#E8E8E8', edgecolor='black', linewidth=2)
            ax.add_patch(summary_rect)

            ax.text(5.5, summary_y - 0.2, 'Disk Capacity Summary', ha='center', fontsize=12, fontweight='bold')
            ax.text(3, summary_y - 0.6, f'• Total Tracks: 35', fontsize=10, ha='left')
            ax.text(3, summary_y - 0.9, f'• Total Sectors: {total_sectors}', fontsize=10, ha='left')
            ax.text(6.5, summary_y - 0.6, f'• Bytes/Sector: 256', fontsize=10, ha='left')
            ax.text(6.5, summary_y - 0.9, f'• Total Capacity: ~170 KB', fontsize=10, ha='left')

            filename = '1541_disk_layout.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': '1541 Disk Track/Sector Layout',
                'description': 'Zone-based track organization and disk capacity breakdown'
            })

        # 6510 I/O Port Registers (unique to 6510, not in 6502)
        if '6510' in title_upper:
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.set_xlim(0, 12)
            ax.set_ylim(0, 9)
            ax.axis('off')

            ax.text(6, 8.5, '6510 I/O Port Registers',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(6, 8, 'Memory Locations $0000 and $0001',
                   ha='center', fontsize=11, style='italic', color='#666666')

            # Data Direction Register ($0000)
            y = 6.5
            rect = FancyBboxPatch((1, y-0.4), 4, 0.8,
                                 boxstyle="round,pad=0.05",
                                 facecolor='#4A90E2', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(3, y, '$0000 - Data Direction Register',
                   ha='center', va='center', fontsize=12, fontweight='bold', color='white')

            # Data Port Register ($0001)
            y = 5.3
            rect = FancyBboxPatch((1, y-0.4), 4, 0.8,
                                 boxstyle="round,pad=0.05",
                                 facecolor='#50C878', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(3, y, '$0001 - Data Port Register',
                   ha='center', va='center', fontsize=12, fontweight='bold', color='white')

            # Bit functions
            functions_y = 3.5
            ax.text(6, functions_y + 0.5, 'Port Bit Functions:', ha='center', fontsize=12, fontweight='bold')

            bit_functions = [
                'Bit 0: Cassette Data Output',
                'Bit 1: Cassette Write',
                'Bit 2: Cassette Motor (0=On)',
                'Bit 3: Cassette Sense',
                'Bit 4: Cassette Read',
                'Bit 5: Bank Select (0=BASIC, 1=I/O)',
                'Bits 0-2: Memory Configuration',
            ]

            for i, func in enumerate(bit_functions):
                row = i // 2
                col = i % 2
                x = 2 if col == 0 else 7.5
                y = functions_y - 0.4 - (row * 0.4)
                ax.text(x, y, f'• {func}', fontsize=9, ha='left')

            # Bank switching note
            note_y = 0.8
            note_rect = FancyBboxPatch((1.5, note_y - 0.5), 9, 0.8,
                                      boxstyle="round,pad=0.1",
                                      facecolor='#FFF3CD', edgecolor='#856404', linewidth=1.5)
            ax.add_patch(note_rect)
            ax.text(6, note_y, 'Controls memory banking: KERNAL ROM, BASIC ROM, I/O, and Character ROM',
                   ha='center', va='center', fontsize=9, style='italic')

            filename = '6510_io_ports.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': '6510 I/O Port Registers',
                'description': 'Memory banking and cassette control ports unique to the 6510 CPU'
            })

        # VIC Chip Register Map (VIC-20, different from VIC-II)
        if title_upper == 'VIC' and 'VIC-II' not in title_upper:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 14)
            ax.axis('off')

            ax.text(5, 13.5, 'VIC Chip Register Map',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(5, 13, 'VIC-20 Video Interface Chip ($9000-$900F)',
                   ha='center', fontsize=11, style='italic', color='#666666')

            y_start = 11.5
            registers = [
                ('$9000', 'Horizontal Center', '#4A90E2'),
                ('$9001', 'Vertical Center', '#4A90E2'),
                ('$9002', 'Columns (bits 0-6)', '#50C878'),
                ('$9003', 'Rows (bits 1-6)', '#50C878'),
                ('$9004', 'Raster Value', '#E76F51'),
                ('$9005', 'Video/Char Memory (bits 0-3)', '#F4A261'),
                ('$9006', 'Light Pen Horizontal', '#9D4EDD'),
                ('$9007', 'Light Pen Vertical', '#9D4EDD'),
                ('$9008', 'Paddle X', '#2A9D8F'),
                ('$9009', 'Paddle Y', '#2A9D8F'),
                ('$900A', 'Bass Sound', '#E63946'),
                ('$900B', 'Alto Sound', '#E63946'),
                ('$900C', 'Soprano Sound', '#E63946'),
                ('$900D', 'Noise Sound', '#E63946'),
                ('$900E', 'Auxiliary Color', '#F4A261'),
                ('$900F', 'Screen/Border/Reverse', '#4A90E2'),
            ]

            for i, (addr, desc, color) in enumerate(registers):
                y = y_start - (i * 0.7)
                rect = FancyBboxPatch((0.5, y-0.3), 3, 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(2, y, addr, ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')
                ax.text(4, y, desc, va='center', fontsize=8.5)

            filename = 'vic_memory_map.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'VIC Chip Register Map',
                'description': 'Register layout for the VIC-20 Video Interface Chip'
            })

        # C64 Memory Map
        if 'MEMORY' in title_upper or title_upper in ['C64', 'COMMODORE 64', 'COMMODORE-64']:
            fig, ax = plt.subplots(figsize=(14, 12))
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 20)
            ax.axis('off')

            ax.text(7, 19.5, 'Commodore 64 Memory Map',
                   ha='center', fontsize=18, fontweight='bold')
            ax.text(7, 19, '64 KB Address Space ($0000-$FFFF)',
                   ha='center', fontsize=12, style='italic', color='#666666')

            # Memory regions with addresses, names, sizes, and colors
            y_start = 17.5
            regions = [
                ('$0000-$00FF', 'Zero Page', '256 bytes', '#E63946'),
                ('$0100-$01FF', 'Stack', '256 bytes', '#E76F51'),
                ('$0200-$03FF', 'BASIC/KERNAL Variables', '512 bytes', '#F4A261'),
                ('$0400-$07FF', 'Screen RAM (default)', '1 KB', '#4A90E2'),
                ('$0800-$9FFF', 'BASIC Program RAM', '38 KB', '#50C878'),
                ('$A000-$BFFF', 'BASIC ROM', '8 KB', '#9D4EDD'),
                ('$C000-$CFFF', 'RAM (under BASIC ROM)', '4 KB', '#50C878'),
                ('$D000-$D3FF', 'VIC-II Registers', '1 KB', '#4A90E2'),
                ('$D400-$D7FF', 'SID Registers', '1 KB', '#E63946'),
                ('$D800-$DBFF', 'Color RAM', '1 KB', '#F4A261'),
                ('$DC00-$DCFF', 'CIA1 Registers', '256 bytes', '#2A9D8F'),
                ('$DD00-$DDFF', 'CIA2 Registers', '256 bytes', '#2A9D8F'),
                ('$DE00-$DFFF', 'I/O Expansion', '512 bytes', '#CCCCCC'),
                ('$E000-$FFFF', 'KERNAL ROM', '8 KB', '#9D4EDD'),
            ]

            bar_height = 0.7
            for i, (addr, name, size, color) in enumerate(regions):
                y = y_start - (i * 0.95)

                # Address box
                addr_rect = FancyBboxPatch((0.5, y-0.35), 3.5, bar_height,
                                          boxstyle="round,pad=0.05",
                                          facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(addr_rect)
                ax.text(2.25, y, addr, ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white')

                # Region name
                ax.text(4.5, y, name, va='center', fontsize=11, fontweight='bold')

                # Size
                ax.text(10, y, size, va='center', fontsize=9,
                       ha='right', style='italic', color='#666666')

            # Add memory banking note
            note_y = 2.5
            note_rect = FancyBboxPatch((0.5, note_y - 1.2), 13, 2,
                                      boxstyle="round,pad=0.15",
                                      facecolor='#FFF3CD', edgecolor='#856404', linewidth=2)
            ax.add_patch(note_rect)

            ax.text(7, note_y + 0.5, 'Memory Banking Notes:', ha='center',
                   fontsize=11, fontweight='bold')

            notes = [
                '• BASIC ROM ($A000-$BFFF) can be switched out to access RAM',
                '• KERNAL ROM ($E000-$FFFF) can be switched out to access RAM',
                '• I/O area ($D000-$DFFF) can be switched to Character ROM or RAM',
                '• Bank switching controlled via 6510 port at $0001',
                '• Total: 64 KB addressable, with banking for ROM/RAM switching',
            ]

            for i, note in enumerate(notes):
                y = note_y + 0.1 - (i * 0.35)
                ax.text(7, y, note, ha='center', fontsize=8.5)

            filename = 'c64_memory_map.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'C64 Memory Map',
                'description': 'Complete 64KB address space layout with ROM, RAM, and I/O regions'
            })

        # Joystick Control Port Pinout
        if 'JOYSTICK' in title_upper or 'CONTROL PORT' in title_upper:
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.set_xlim(0, 12)
            ax.set_ylim(0, 14)
            ax.axis('off')

            ax.text(6, 13, 'Joystick Control Ports',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(6, 12.5, 'CIA1: Port A ($DC00) = Port 2  |  CIA1: Port B ($DC01) = Port 1',
                   ha='center', fontsize=11, style='italic', color='#666666')

            # Port 1 and Port 2 pinouts
            y_start = 11
            pins = [
                ('Pin 1', 'Up', '#4A90E2', 'Bit 0'),
                ('Pin 2', 'Down', '#4A90E2', 'Bit 1'),
                ('Pin 3', 'Left', '#4A90E2', 'Bit 2'),
                ('Pin 4', 'Right', '#4A90E2', 'Bit 3'),
                ('Pin 5', 'Paddle Y', '#F4A261', 'Analog'),
                ('Pin 6', 'Fire Button', '#E63946', 'Bit 4'),
                ('Pin 7', '+5V Power', '#50C878', 'Power'),
                ('Pin 8', 'Ground', '#666666', 'GND'),
                ('Pin 9', 'Paddle X', '#F4A261', 'Analog'),
            ]

            for i, (pin, function, color, note) in enumerate(pins):
                y = y_start - (i * 0.8)
                rect = FancyBboxPatch((1, y-0.35), 4, 0.6,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(3, y, f'{pin}: {function}', ha='center', va='center',
                       fontsize=11, fontweight='bold', color='white')
                ax.text(5.5, y, note, va='center', fontsize=9, style='italic')

            # Reading joystick code example
            note_y = 2.5
            note_rect = FancyBboxPatch((1, note_y - 0.8), 10, 1.5,
                                      boxstyle="round,pad=0.1",
                                      facecolor='#F0F0F0', edgecolor='#333333', linewidth=1.5)
            ax.add_patch(note_rect)
            ax.text(6, note_y + 0.3, 'Reading Joystick (Port 2):', ha='center', fontsize=10, fontweight='bold')
            ax.text(6, note_y - 0.1, 'LDA $DC00  ; Read CIA1 Port A', ha='center', fontsize=9, family='monospace')
            ax.text(6, note_y - 0.4, 'Bit = 0 means pressed, Bit = 1 means released', ha='center', fontsize=9, style='italic')

            filename = 'joystick_pinout.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'Joystick Control Port Pinout',
                'description': '9-pin D-sub connector pinout for C64 joystick ports'
            })

        # Keyboard Matrix Diagram
        if 'KEYBOARD' in title_upper:
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 12)
            ax.axis('off')

            ax.text(7, 11.5, 'C64 Keyboard Matrix (8x8)',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(7, 11, 'CIA1: Port A ($DC00) = Rows  |  CIA1: Port B ($DC01) = Columns',
                   ha='center', fontsize=11, style='italic', color='#666666')

            # Simplified keyboard matrix showing structure
            y_start = 9.5
            rows = [
                'Row 0: DELETE, RETURN, →, F7, F1, F3, F5, ↓',
                'Row 1: 3, W, A, 4, Z, S, E, Shift(L)',
                'Row 2: 5, R, D, 6, C, F, T, X',
                'Row 3: 7, Y, G, 8, B, H, U, V',
                'Row 4: 9, I, J, 0, M, K, O, N',
                'Row 5: +, P, L, -, ., :, @, ,',
                'Row 6: £, *, ;, HOME, Shift(R), =, ↑, /',
                'Row 7: 1, ←, CTRL, 2, SPACE, C=, Q, RUN/STOP',
            ]

            colors = ['#4A90E2', '#50C878', '#E76F51', '#9D4EDD', '#F4A261', '#2A9D8F', '#E63946', '#4ECDC4']

            for i, (row_text, color) in enumerate(zip(rows, colors)):
                y = y_start - (i * 0.9)
                rect = FancyBboxPatch((0.5, y-0.35), 13, 0.6,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(7, y, row_text, ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')

            # Scanning example
            note_y = 1.5
            note_rect = FancyBboxPatch((1, note_y - 0.6), 12, 1.2,
                                      boxstyle="round,pad=0.1",
                                      facecolor='#F0F0F0', edgecolor='#333333', linewidth=1.5)
            ax.add_patch(note_rect)
            ax.text(7, note_y + 0.2, 'Scanning: Set row low on $DC00, read columns from $DC01',
                   ha='center', fontsize=9, fontweight='bold')
            ax.text(7, note_y - 0.2, 'Bit = 0 means key pressed, Bit = 1 means key released',
                   ha='center', fontsize=9, style='italic')

            filename = 'keyboard_matrix.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'Keyboard Matrix Layout',
                'description': '8x8 keyboard matrix showing key positions and scanning method'
            })

        # PETSCII Character Codes
        if 'PETSCII' in title_upper or 'CHARACTER CODE' in title_upper:
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 12)
            ax.axis('off')

            ax.text(7, 11.5, 'PETSCII Character Codes',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(7, 11, 'PET Standard Code of Information Interchange',
                   ha='center', fontsize=11, style='italic', color='#666666')

            # Key character ranges
            y_start = 9.5
            ranges = [
                ('$00-$1F', 'Control Characters', '#E63946', '32 codes: cursor, color, clear screen'),
                ('$20-$3F', 'Uppercase + Symbols', '#4A90E2', '32 codes: SPACE ! " # $ % & \' ( ) * + , - . /'),
                ('$40-$5F', 'Uppercase Letters', '#50C878', '32 codes: @ A-Z [ \\ ] ↑ ←'),
                ('$60-$7F', 'Lowercase + Graphics', '#9D4EDD', '32 codes: graphic symbols and lowercase a-z'),
                ('$80-$9F', 'Control Characters (Reverse)', '#E76F51', '32 codes: same as $00-$1F but reversed'),
                ('$A0-$BF', 'Uppercase + Symbols (Reverse)', '#2A9D8F', '32 codes: reversed versions of $20-$3F'),
                ('$C0-$DF', 'Graphics Characters', '#F4A261', '32 codes: block graphics and symbols'),
                ('$E0-$FF', 'Lowercase + Graphics (Reverse)', '#4ECDC4', '32 codes: reversed versions of $60-$7F'),
            ]

            for i, (range_hex, name, color, desc) in enumerate(ranges):
                y = y_start - (i * 0.9)
                rect = FancyBboxPatch((0.5, y-0.35), 3, 0.6,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(2, y, range_hex, ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white')
                ax.text(4, y + 0.1, name, va='center', fontsize=10, fontweight='bold')
                ax.text(4, y - 0.2, desc, va='center', fontsize=8, style='italic')

            # Common codes note
            note_y = 1.5
            note_rect = FancyBboxPatch((1, note_y - 0.8), 12, 1.5,
                                      boxstyle="round,pad=0.1",
                                      facecolor='#FFF3CD', edgecolor='#856404', linewidth=1.5)
            ax.add_patch(note_rect)
            ax.text(7, note_y + 0.3, 'Common PETSCII Codes:', ha='center', fontsize=10, fontweight='bold')
            ax.text(7, note_y, '$13=HOME  $14=DEL  $93=CLR  $05=WHITE  $1C=RED  $9E=YELLOW  $1E=GREEN  $1F=BLUE',
                   ha='center', fontsize=9, family='monospace')
            ax.text(7, note_y - 0.4, 'Screen codes differ from PETSCII: $41-$5A (A-Z) → screen codes $01-$1A',
                   ha='center', fontsize=9, style='italic')

            filename = 'petscii_codes.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'PETSCII Character Code Table',
                'description': 'Character encoding ranges and common control codes'
            })

        # C64 Color Palette
        if 'COLOR' in title_upper or 'COLOUR' in title_upper:
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 12)
            ax.axis('off')

            ax.text(7, 11.5, 'Commodore 64 Color Palette',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(7, 11, '16 Colors - VIC-II Color Values',
                   ha='center', fontsize=11, style='italic', color='#666666')

            # C64 color palette (actual C64 colors)
            colors_c64 = [
                (0, 'Black', '#000000'),
                (1, 'White', '#FFFFFF'),
                (2, 'Red', '#880000'),
                (3, 'Cyan', '#AAFFEE'),
                (4, 'Purple', '#CC44CC'),
                (5, 'Green', '#00CC55'),
                (6, 'Blue', '#0000AA'),
                (7, 'Yellow', '#EEEE77'),
                (8, 'Orange', '#DD8855'),
                (9, 'Brown', '#664400'),
                (10, 'Light Red', '#FF7777'),
                (11, 'Dark Grey', '#333333'),
                (12, 'Grey', '#777777'),
                (13, 'Light Green', '#AAFF66'),
                (14, 'Light Blue', '#0088FF'),
                (15, 'Light Grey', '#BBBBBB'),
            ]

            # Display colors in 4x4 grid
            x_start = 1.5
            y_start = 9.5
            cell_width = 2.8
            cell_height = 1.2

            for i, (num, name, hex_color) in enumerate(colors_c64):
                row = i // 4
                col = i % 4
                x = x_start + (col * cell_width)
                y = y_start - (row * cell_height)

                # Color box
                rect = FancyBboxPatch((x, y-0.5), cell_width-0.2, 0.9,
                                     boxstyle="round,pad=0.05",
                                     facecolor=hex_color, edgecolor='black', linewidth=2)
                ax.add_patch(rect)

                # Text color (white or black depending on background)
                text_color = 'white' if num in [0, 2, 6, 9, 11] else 'black'
                ax.text(x + cell_width/2 - 0.1, y, f'{num}: {name}',
                       ha='center', va='center', fontsize=10, fontweight='bold', color=text_color)
                ax.text(x + cell_width/2 - 0.1, y - 0.25, hex_color,
                       ha='center', va='center', fontsize=8, family='monospace', color=text_color)

            # Memory addresses note
            note_y = 2.5
            note_rect = FancyBboxPatch((1, note_y - 0.8), 12, 1.5,
                                      boxstyle="round,pad=0.1",
                                      facecolor='#F0F0F0', edgecolor='#333333', linewidth=1.5)
            ax.add_patch(note_rect)
            ax.text(7, note_y + 0.3, 'Color Registers:', ha='center', fontsize=10, fontweight='bold')
            ax.text(7, note_y, '$D020=Border  $D021=Background  $D800-$DBFF=Color RAM (screen color, 1000 bytes)',
                   ha='center', fontsize=9, family='monospace')
            ax.text(7, note_y - 0.4, 'POKE 53280,0: Black border  |  POKE 53281,1: White background',
                   ha='center', fontsize=9, style='italic')

            filename = 'color_palette.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'C64 Color Palette',
                'description': 'All 16 VIC-II colors with hex values and memory addresses'
            })

        # Interrupt Vectors and Timing
        if 'INTERRUPT' in title_upper or 'IRQ' in title_upper or 'NMI' in title_upper:
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.set_xlim(0, 12)
            ax.set_ylim(0, 14)
            ax.axis('off')

            ax.text(6, 13, 'C64 Interrupt System',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(6, 12.5, '6510 CPU Interrupt Vectors and Handling',
                   ha='center', fontsize=11, style='italic', color='#666666')

            # Interrupt vectors
            y_start = 11
            vectors = [
                ('$FFFE-$FFFF', 'IRQ Vector (Hardware)', '#4A90E2', 'Default: $FF48 (KERNAL IRQ handler)'),
                ('$0314-$0315', 'IRQ Vector (RAM)', '#50C878', 'User IRQ vector (redirected by KERNAL)'),
                ('$FFFA-$FFFB', 'NMI Vector (Hardware)', '#E63946', 'Default: $FE43 (KERNAL NMI handler)'),
                ('$0318-$0319', 'NMI Vector (RAM)', '#E76F51', 'User NMI vector (redirected by KERNAL)'),
            ]

            for i, (addr, name, color, desc) in enumerate(vectors):
                y = y_start - (i * 1.2)
                rect = FancyBboxPatch((1, y-0.5), 4, 0.9,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(3, y - 0.1, addr, ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white', family='monospace')
                ax.text(3, y + 0.2, name, ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')
                ax.text(5.5, y, desc, va='center', fontsize=8, style='italic')

            # IRQ sources
            y_irq = 5.5
            ax.text(6, y_irq + 0.5, 'IRQ Sources (Maskable):', ha='center', fontsize=11, fontweight='bold')

            irq_sources = [
                'VIC-II Raster IRQ ($D012 comparison)',
                'CIA1 Timer A/B ($DC0D)',
                'CIA2 Timer A/B ($DD0D)',
            ]

            for i, source in enumerate(irq_sources):
                y = y_irq - (i * 0.5)
                ax.text(6, y, f'• {source}', ha='center', fontsize=9)

            # Raster interrupt example
            note_y = 2.5
            note_rect = FancyBboxPatch((1, note_y - 1.2), 10, 2.2,
                                      boxstyle="round,pad=0.1",
                                      facecolor='#F0F0F0', edgecolor='#333333', linewidth=1.5)
            ax.add_patch(note_rect)
            ax.text(6, note_y + 0.7, 'Setting Up Raster IRQ:', ha='center', fontsize=10, fontweight='bold')
            ax.text(6, note_y + 0.3, 'LDA #$7F : STA $DC0D  ; Disable CIA interrupts',
                   ha='center', fontsize=8, family='monospace')
            ax.text(6, note_y, 'LDA #$01 : STA $D01A  ; Enable raster IRQ',
                   ha='center', fontsize=8, family='monospace')
            ax.text(6, note_y - 0.3, 'LDA #$80 : STA $D012  ; Set raster line to 128',
                   ha='center', fontsize=8, family='monospace')
            ax.text(6, note_y - 0.6, 'LDA #<IRQ : STA $0314  ; Set IRQ vector low byte',
                   ha='center', fontsize=8, family='monospace')
            ax.text(6, note_y - 0.9, 'LDA #>IRQ : STA $0315  ; Set IRQ vector high byte',
                   ha='center', fontsize=8, family='monospace')

            filename = 'interrupt_vectors.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'Interrupt Vectors and Setup',
                'description': 'IRQ/NMI vectors, sources, and raster interrupt configuration'
            })

        # Character Set Overview
        if 'CHARACTER SET' in title_upper or 'CHARSET' in title_upper or 'FONT' in title_upper:
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.set_xlim(0, 12)
            ax.set_ylim(0, 14)
            ax.axis('off')

            ax.text(6, 13, 'C64 Character Sets',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(6, 12.5, 'ROM Character Generator and Custom Fonts',
                   ha='center', fontsize=11, style='italic', color='#666666')

            # Character ROM locations
            y_start = 11
            charsets = [
                ('$D000-$D7FF', 'Uppercase/Graphics ROM', '#4A90E2', '2KB: UPPERCASE letters + graphics'),
                ('$D800-$DFFF', 'Lowercase/Uppercase ROM', '#50C878', '2KB: lowercase + UPPERCASE letters'),
            ]

            for i, (addr, name, color, desc) in enumerate(charsets):
                y = y_start - (i * 1.2)
                rect = FancyBboxPatch((1, y-0.5), 4, 0.9,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(3, y - 0.1, addr, ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white', family='monospace')
                ax.text(3, y + 0.2, name, ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')
                ax.text(5.5, y, desc, va='center', fontsize=8, style='italic')

            # Character format
            y_format = 8
            ax.text(6, y_format, 'Character Format: 8x8 pixels, 8 bytes per character',
                   ha='center', fontsize=10, fontweight='bold')

            # Custom character locations
            y_custom = 6.5
            ax.text(6, y_custom + 0.5, 'Custom Character Sets (User-Defined):', ha='center', fontsize=11, fontweight='bold')

            custom_locs = [
                'Can be placed in RAM at any 2KB boundary',
                'VIC-II bank selection via $DD00 (CIA2)',
                'Character memory pointer via $D018',
                'Total 256 characters (8 bytes each = 2048 bytes)',
            ]

            for i, loc in enumerate(custom_locs):
                y = y_custom - (i * 0.5)
                ax.text(6, y, f'• {loc}', ha='center', fontsize=9)

            # Switching example
            note_y = 2.5
            note_rect = FancyBboxPatch((1, note_y - 1), 10, 1.8,
                                      boxstyle="round,pad=0.1",
                                      facecolor='#F0F0F0', edgecolor='#333333', linewidth=1.5)
            ax.add_patch(note_rect)
            ax.text(6, note_y + 0.5, 'Switching to Lowercase/Uppercase:', ha='center', fontsize=10, fontweight='bold')
            ax.text(6, note_y + 0.1, 'POKE 53272,23  ; $D018=$17 (lowercase mode)',
                   ha='center', fontsize=9, family='monospace')
            ax.text(6, note_y - 0.2, 'POKE 53272,21  ; $D018=$15 (uppercase/graphics mode)',
                   ha='center', fontsize=9, family='monospace')
            ax.text(6, note_y - 0.6, 'Each character = 8 bytes: bit 7=leftmost pixel, bit 0=rightmost pixel',
                   ha='center', fontsize=9, style='italic')

            filename = 'character_set.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'Character Set Layout',
                'description': 'ROM character sets and custom font configuration'
            })

        # ADSR Envelope Diagram
        if 'ADSR' in title_upper or ('SOUND' in title_upper and 'ENVELOPE' in title_upper):
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 12)
            ax.axis('off')

            ax.text(7, 11.5, 'SID ADSR Envelope',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(7, 11, 'Attack - Decay - Sustain - Release',
                   ha='center', fontsize=11, style='italic', color='#666666')

            # Draw ADSR envelope shape
            import numpy as np

            # Time points for ADSR phases
            attack_time = 2
            decay_time = 1.5
            sustain_time = 3
            release_time = 2

            # Amplitude values
            peak_level = 7
            sustain_level = 5

            # Create envelope curve
            x_vals = []
            y_vals = []

            # Attack phase (0 to peak)
            for i in range(20):
                t = i / 20 * attack_time
                x_vals.append(1 + t)
                y_vals.append(3 + (i / 20) * peak_level)

            # Decay phase (peak to sustain)
            for i in range(15):
                t = i / 15 * decay_time
                x_vals.append(1 + attack_time + t)
                y_vals.append(3 + peak_level - (i / 15) * (peak_level - sustain_level))

            # Sustain phase (hold at sustain level)
            for i in range(30):
                t = i / 30 * sustain_time
                x_vals.append(1 + attack_time + decay_time + t)
                y_vals.append(3 + sustain_level)

            # Release phase (sustain to 0)
            for i in range(20):
                t = i / 20 * release_time
                x_vals.append(1 + attack_time + decay_time + sustain_time + t)
                y_vals.append(3 + sustain_level - (i / 20) * sustain_level)

            # Plot envelope
            ax.plot(x_vals, y_vals, color='#4A90E2', linewidth=3)
            ax.fill_between(x_vals, 3, y_vals, alpha=0.3, color='#4A90E2')

            # Label phases
            phase_labels = [
                (1 + attack_time/2, 10.5, 'ATTACK', '#E63946'),
                (1 + attack_time + decay_time/2, 10.5, 'DECAY', '#F4A261'),
                (1 + attack_time + decay_time + sustain_time/2, 10.5, 'SUSTAIN', '#50C878'),
                (1 + attack_time + decay_time + sustain_time + release_time/2, 10.5, 'RELEASE', '#9D4EDD'),
            ]

            for x, y, label, color in phase_labels:
                ax.text(x, y, label, ha='center', fontsize=10, fontweight='bold', color=color)
                ax.axvline(x, ymin=0.25, ymax=0.87, linestyle='--', alpha=0.3, color=color)

            # SID registers
            y_reg = 1.5
            ax.text(7, y_reg + 0.5, 'SID ADSR Control Registers:', ha='center', fontsize=11, fontweight='bold')

            registers = [
                ('$D405/$D40C/$D413', 'Attack/Decay', 'High nibble = Attack, Low nibble = Decay'),
                ('$D406/$D40D/$D414', 'Sustain/Release', 'High nibble = Sustain, Low nibble = Release'),
            ]

            for i, (addr, name, desc) in enumerate(registers):
                y = y_reg - (i * 0.4)
                ax.text(7, y, f'{addr}: {name} - {desc}', ha='center', fontsize=9, family='monospace')

            filename = 'adsr_envelope.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'ADSR Envelope',
                'description': 'SID sound envelope visualization with Attack, Decay, Sustain, Release phases'
            })

        # Bitmap Mode Memory Layout
        if 'BITMAP' in title_upper or 'HIRES' in title_upper:
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.set_xlim(0, 12)
            ax.set_ylim(0, 14)
            ax.axis('off')

            ax.text(6, 13, 'C64 Bitmap Mode Memory Layout',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(6, 12.5, 'Hi-Resolution 320x200 (8000 bytes bitmap + 1000 bytes color)',
                   ha='center', fontsize=11, style='italic', color='#666666')

            # Bitmap memory
            y_start = 11
            bitmap_sections = [
                ('Bitmap Data', '8000 bytes', '$2000-$3FFF (typical)', '#4A90E2'),
                ('320 x 200 pixels', '40 x 25 chars', '8 bytes per character', '#4A90E2'),
            ]

            rect = FancyBboxPatch((1, y_start - 0.8), 10, 1.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#4A90E2', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(6, y_start - 0.2, 'Bitmap Data (8000 bytes)', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white')
            ax.text(6, y_start - 0.6, '$2000-$3FFF (typical location)', ha='center', va='center',
                   fontsize=10, color='white')

            # Color RAM
            y_color = 8.5
            rect = FancyBboxPatch((1, y_color - 0.5), 10, 0.9,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#50C878', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(6, y_color, 'Screen RAM (1000 bytes) - $0400-$07E7', ha='center', va='center',
                   fontsize=11, fontweight='bold', color='white')
            ax.text(6, y_color - 0.3, 'High nibble = foreground color, Low nibble = background color',
                   ha='center', va='center', fontsize=9, color='white')

            # Pixel format
            y_pixel = 6.5
            ax.text(6, y_pixel + 0.5, 'Pixel Format (Each Character Cell = 8x8 pixels):', ha='center', fontsize=11, fontweight='bold')

            pixel_info = [
                'Each byte = 8 pixels (1 bit per pixel)',
                'Bit 7 = leftmost pixel, Bit 0 = rightmost pixel',
                '1 = foreground color, 0 = background color',
                'Total: 40 chars wide x 25 chars high = 1000 character cells',
            ]

            for i, info in enumerate(pixel_info):
                y = y_pixel - (i * 0.4)
                ax.text(6, y, f'• {info}', ha='center', fontsize=9)

            # VIC-II configuration
            note_y = 3.5
            note_rect = FancyBboxPatch((1, note_y - 1.2), 10, 2.2,
                                      boxstyle="round,pad=0.1",
                                      facecolor='#F0F0F0', edgecolor='#333333', linewidth=1.5)
            ax.add_patch(note_rect)
            ax.text(6, note_y + 0.7, 'Enabling Bitmap Mode:', ha='center', fontsize=10, fontweight='bold')
            ax.text(6, note_y + 0.3, 'LDA $D011 : ORA #$20 : STA $D011  ; Set bit 5 = bitmap mode',
                   ha='center', fontsize=9, family='monospace')
            ax.text(6, note_y - 0.1, 'LDA $D018 : ORA #$08 : STA $D018  ; Set bitmap at $2000',
                   ha='center', fontsize=9, family='monospace')
            ax.text(6, note_y - 0.5, 'BASIC: POKE 53265,PEEK(53265) OR 32',
                   ha='center', fontsize=9, style='italic')
            ax.text(6, note_y - 0.9, 'BASIC: POKE 53272,PEEK(53272) OR 8',
                   ha='center', fontsize=9, style='italic')

            filename = 'bitmap_mode.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'Bitmap Mode Layout',
                'description': 'Hi-res bitmap mode memory organization and VIC-II configuration'
            })

        # Screen Memory Layout
        if 'SCREEN' in title_upper and 'BITMAP' not in title_upper:
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.set_xlim(0, 12)
            ax.set_ylim(0, 14)
            ax.axis('off')

            ax.text(6, 13, 'C64 Screen Memory Layout',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(6, 12.5, 'Character Mode (40x25 = 1000 bytes)',
                   ha='center', fontsize=11, style='italic', color='#666666')

            # Screen RAM
            y_start = 11
            rect = FancyBboxPatch((1, y_start - 0.8), 10, 1.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#4A90E2', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(6, y_start - 0.2, 'Screen RAM (1000 bytes)', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white')
            ax.text(6, y_start - 0.6, '$0400-$07E7 (default)', ha='center', va='center',
                   fontsize=10, color='white')

            # Color RAM
            y_color = 8.8
            rect = FancyBboxPatch((1, y_color - 0.5), 10, 0.9,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#50C878', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(6, y_color, 'Color RAM (1000 bytes) - $D800-$DBE7', ha='center', va='center',
                   fontsize=11, fontweight='bold', color='white')
            ax.text(6, y_color - 0.3, 'Character color (low nibble only, 0-15)',
                   ha='center', va='center', fontsize=9, color='white')

            # Screen layout
            y_layout = 7
            ax.text(6, y_layout + 0.5, 'Screen Organization:', ha='center', fontsize=11, fontweight='bold')

            layout_info = [
                '40 columns x 25 rows = 1000 character positions',
                'Each byte = character code (0-255)',
                'Row 0: $0400-$0427 (bytes 0-39)',
                'Row 1: $0428-$044F (bytes 40-79)',
                '...',
                'Row 24: $07C0-$07E7 (bytes 960-999)',
            ]

            for i, info in enumerate(layout_info):
                y = y_layout - (i * 0.35)
                ax.text(6, y, info, ha='center', fontsize=9)

            # Addressing formula
            note_y = 3
            note_rect = FancyBboxPatch((1, note_y - 1), 10, 1.8,
                                      boxstyle="round,pad=0.1",
                                      facecolor='#F0F0F0', edgecolor='#333333', linewidth=1.5)
            ax.add_patch(note_rect)
            ax.text(6, note_y + 0.5, 'Screen Position Formula:', ha='center', fontsize=10, fontweight='bold')
            ax.text(6, note_y + 0.1, 'Address = $0400 + (Row * 40) + Column',
                   ha='center', fontsize=9, family='monospace')
            ax.text(6, note_y - 0.3, 'Example: Row 5, Column 10 = $0400 + (5*40) + 10 = $04CA',
                   ha='center', fontsize=9, style='italic')
            ax.text(6, note_y - 0.7, 'BASIC: POKE 1024 + (Row*40) + Column, CharCode',
                   ha='center', fontsize=9, style='italic')

            filename = 'screen_layout.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'Screen Memory Layout',
                'description': 'Character mode screen and color RAM organization with addressing'
            })

        # Raster Beam Timing Diagram
        if 'RASTER' in title_upper:
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 12)
            ax.axis('off')

            ax.text(7, 11.5, 'C64 Raster Beam Timing',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(7, 11, 'VIC-II Raster Scan: 312 lines (PAL) / 263 lines (NTSC)',
                   ha='center', fontsize=11, style='italic', color='#666666')

            # Draw raster lines representation
            y_start = 9.5

            # Visible area
            visible_rect = FancyBboxPatch((2, y_start - 4), 10, 3.5,
                                         boxstyle="round,pad=0.05",
                                         facecolor='#E8F4F8', edgecolor='#4A90E2', linewidth=2)
            ax.add_patch(visible_rect)
            ax.text(7, y_start - 2.2, 'Visible Area', ha='center', fontsize=12, fontweight='bold', color='#4A90E2')
            ax.text(7, y_start - 2.6, '200 raster lines (lines 51-250 PAL)', ha='center', fontsize=9)

            # Border areas
            top_border = FancyBboxPatch((2, y_start - 0.8), 10, 0.7,
                                       boxstyle="round,pad=0.05",
                                       facecolor='#F0F0F0', edgecolor='#666666', linewidth=1)
            ax.add_patch(top_border)
            ax.text(7, y_start - 0.45, 'Top Border', ha='center', fontsize=9, color='#666666')

            bottom_border = FancyBboxPatch((2, y_start - 4.8), 10, 0.7,
                                          boxstyle="round,pad=0.05",
                                          facecolor='#F0F0F0', edgecolor='#666666', linewidth=1)
            ax.add_patch(bottom_border)
            ax.text(7, y_start - 4.45, 'Bottom Border', ha='center', fontsize=9, color='#666666')

            # Raster register info
            y_reg = 4
            ax.text(7, y_reg + 0.5, 'Raster Line Register ($D012):', ha='center', fontsize=11, fontweight='bold')

            raster_info = [
                'Read current raster line: LDA $D012',
                'Set raster interrupt: STA $D012 (0-255)',
                'Line 0 = top of screen (in border area)',
                'Lines 51-250 = visible display area (PAL)',
                'Use with $D011 bit 7 for lines 256-312',
            ]

            for i, info in enumerate(raster_info):
                y = y_reg - (i * 0.35)
                ax.text(7, y, f'• {info}', ha='center', fontsize=9)

            # Raster interrupt example
            note_y = 1.2
            note_rect = FancyBboxPatch((1, note_y - 0.8), 12, 1.5,
                                      boxstyle="round,pad=0.1",
                                      facecolor='#F0F0F0', edgecolor='#333333', linewidth=1.5)
            ax.add_patch(note_rect)
            ax.text(7, note_y + 0.3, 'Raster Split Example (change border color at line 128):', ha='center', fontsize=10, fontweight='bold')
            ax.text(7, note_y - 0.1, 'LDA #128 : STA $D012  ; Trigger at line 128',
                   ha='center', fontsize=8, family='monospace')
            ax.text(7, note_y - 0.4, 'IRQ: INC $D020 : ASL $D019 : RTI  ; Change border, acknowledge, return',
                   ha='center', fontsize=8, family='monospace')

            filename = 'raster_timing.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'Raster Beam Timing',
                'description': 'VIC-II raster scan timing, line numbering, and interrupt usage'
            })

        # Multicolor Mode Pixel Layout
        if 'MULTICOLOR' in title_upper or 'MULTI COLOR' in title_upper or 'MULTI-COLOR' in title_upper:
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.set_xlim(0, 12)
            ax.set_ylim(0, 14)
            ax.axis('off')

            ax.text(6, 13, 'C64 Multicolor Mode',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(6, 12.5, 'Character Mode: 4 colors, 160x200 resolution (4x8 pixels per char)',
                   ha='center', fontsize=11, style='italic', color='#666666')

            # Pixel bit pattern
            y_start = 11
            ax.text(6, y_start, 'Bit Pair Encoding (2 bits per pixel):', ha='center', fontsize=11, fontweight='bold')

            # Create visual representation of bit pairs
            bit_patterns = [
                ('00', 'Background ($D021)', '#000000', 'white'),
                ('01', 'Upper Color RAM', '#FF0000', 'white'),
                ('10', 'Lower Color RAM', '#00FF00', 'black'),
                ('11', 'Character Color', '#0000FF', 'white'),
            ]

            y_bits = y_start - 1
            for i, (bits, desc, bg_color, text_color) in enumerate(bit_patterns):
                y = y_bits - (i * 0.8)

                # Bit pattern box
                rect = FancyBboxPatch((2, y - 0.3), 2, 0.6,
                                     boxstyle="round,pad=0.05",
                                     facecolor='#F0F0F0', edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(3, y, bits, ha='center', va='center',
                       fontsize=11, fontweight='bold', family='monospace')

                # Color preview
                color_rect = FancyBboxPatch((4.5, y - 0.3), 1, 0.6,
                                           boxstyle="round,pad=0.05",
                                           facecolor=bg_color, edgecolor='black', linewidth=1.5)
                ax.add_patch(color_rect)

                # Description
                ax.text(6, y, f'→ {desc}', va='center', fontsize=9)

            # Character layout
            y_char = 6
            ax.text(6, y_char + 0.5, 'Character Cell Layout:', ha='center', fontsize=11, fontweight='bold')

            char_info = [
                'Each byte = 4 pixels (2 bits per pixel)',
                'Resolution: 4 pixels wide x 8 pixels high',
                'Screen: 160x200 effective pixels',
                'Characters appear "wider" than hi-res mode',
            ]

            for i, info in enumerate(char_info):
                y = y_char - (i * 0.35)
                ax.text(6, y, f'• {info}', ha='center', fontsize=9)

            # Enabling multicolor
            note_y = 3
            note_rect = FancyBboxPatch((1, note_y - 1.2), 10, 2.2,
                                      boxstyle="round,pad=0.1",
                                      facecolor='#F0F0F0', edgecolor='#333333', linewidth=1.5)
            ax.add_patch(note_rect)
            ax.text(6, note_y + 0.7, 'Enabling Multicolor Character Mode:', ha='center', fontsize=10, fontweight='bold')
            ax.text(6, note_y + 0.3, 'LDA $D016 : ORA #$10 : STA $D016  ; Enable multicolor mode',
                   ha='center', fontsize=9, family='monospace')
            ax.text(6, note_y - 0.1, 'Set Color RAM high bit (>127) for each multicolor character',
                   ha='center', fontsize=9, style='italic')
            ax.text(6, note_y - 0.5, 'BASIC: POKE 53270,PEEK(53270) OR 16',
                   ha='center', fontsize=9, style='italic')
            ax.text(6, note_y - 0.9, 'BASIC: POKE 55296+POS,128+COLOR (for each multicolor char)',
                   ha='center', fontsize=9, style='italic')

            filename = 'multicolor_mode.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'Multicolor Mode',
                'description': 'Multicolor character mode pixel encoding and color sources'
            })

        # BASIC Memory Map
        if title_upper == 'BASIC' or 'BASIC PROGRAM' in title_upper:
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.set_xlim(0, 12)
            ax.set_ylim(0, 14)
            ax.axis('off')

            ax.text(6, 13, 'BASIC Program Memory Layout',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(6, 12.5, 'Commodore 64 BASIC V2',
                   ha='center', fontsize=11, style='italic', color='#666666')

            # Memory regions
            y_start = 11
            regions = [
                ('$0800-$9FFF', 'BASIC Program Area', '38 KB', '#50C878'),
                ('$0800-$0801', 'Start of BASIC (pointer)', '2 bytes', '#4A90E2'),
                ('$002B-$002C', 'Start of Variables', 'pointer', '#F4A261'),
                ('$002D-$002E', 'Start of Arrays', 'pointer', '#E76F51'),
                ('$002F-$0030', 'End of Arrays', 'pointer', '#9D4EDD'),
                ('$0031-$0032', 'String Storage', 'pointer', '#2A9D8F'),
            ]

            for i, (addr, name, size, color) in enumerate(regions):
                y = y_start - (i * 0.9)

                if i == 0:  # Main program area
                    rect = FancyBboxPatch((1, y - 0.4), 10, 0.8,
                                         boxstyle="round,pad=0.05",
                                         facecolor=color, edgecolor='black', linewidth=2)
                else:  # Pointers
                    rect = FancyBboxPatch((1, y - 0.3), 10, 0.6,
                                         boxstyle="round,pad=0.05",
                                         facecolor=color, edgecolor='black', linewidth=1.5)

                ax.add_patch(rect)
                ax.text(6, y, f'{addr}: {name} ({size})', ha='center', va='center',
                       fontsize=10 if i == 0 else 9, fontweight='bold', color='white')

            # BASIC pointers
            y_info = 4
            ax.text(6, y_info + 0.5, 'Important BASIC Pointers:', ha='center', fontsize=11, fontweight='bold')

            pointer_info = [
                '$002B-$002C (43-44): Start of variables',
                '$002D-$002E (45-46): Start of arrays',
                '$0037-$0038 (55-56): Bottom of string space',
                '$0039-$003A (57-58): Top of memory for BASIC',
            ]

            for i, info in enumerate(pointer_info):
                y = y_info - (i * 0.35)
                ax.text(6, y, info, ha='center', fontsize=9, family='monospace')

            # BASIC commands
            note_y = 1.5
            note_rect = FancyBboxPatch((1, note_y - 0.8), 10, 1.5,
                                      boxstyle="round,pad=0.1",
                                      facecolor='#F0F0F0', edgecolor='#333333', linewidth=1.5)
            ax.add_patch(note_rect)
            ax.text(6, note_y + 0.3, 'Useful BASIC Commands:', ha='center', fontsize=10, fontweight='bold')
            ax.text(6, note_y - 0.1, 'FRE(0) - Returns free memory in bytes',
                   ha='center', fontsize=9, family='monospace')
            ax.text(6, note_y - 0.4, 'CLR - Clears all variables',
                   ha='center', fontsize=9, family='monospace')

            filename = 'basic_memory.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'BASIC Memory Layout',
                'description': 'BASIC program area and variable storage pointers'
            })

        # Kernal Jump Table
        if 'KERNAL' in title_upper or 'KERNEL' in title_upper:
            fig, ax = plt.subplots(figsize=(14, 11))
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 13)
            ax.axis('off')

            ax.text(7, 12.5, 'KERNAL Jump Table',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(7, 12, 'ROM Routines ($FF81-$FFF5)',
                   ha='center', fontsize=11, style='italic', color='#666666')

            # Common KERNAL routines
            y_start = 11
            routines = [
                ('$FFD2', 'CHROUT', 'Output character to screen', '#4A90E2'),
                ('$FFE4', 'GETIN', 'Get character from keyboard', '#50C878'),
                ('$FFCF', 'CHRIN', 'Input character from channel', '#E76F51'),
                ('$FFD5', 'LOAD', 'Load file from device', '#9D4EDD'),
                ('$FFD8', 'SAVE', 'Save file to device', '#F4A261'),
                ('$FFBA', 'SETLFS', 'Set logical file parameters', '#2A9D8F'),
                ('$FFBD', 'SETNAM', 'Set filename parameters', '#4ECDC4'),
                ('$FFC0', 'OPEN', 'Open logical file', '#E63946'),
                ('$FFC3', 'CLOSE', 'Close logical file', '#E76F51'),
                ('$FFC6', 'CHKIN', 'Set input channel', '#4A90E2'),
                ('$FFC9', 'CHKOUT', 'Set output channel', '#50C878'),
                ('$FFCC', 'CLRCHN', 'Clear I/O channels', '#9D4EDD'),
            ]

            for i, (addr, name, desc, color) in enumerate(routines):
                row = i // 2
                col = i % 2
                x = 1.5 + (col * 5.5)
                y = y_start - (row * 0.65)

                rect = FancyBboxPatch((x, y - 0.25), 5, 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                ax.text(x + 0.8, y, addr, va='center',
                       fontsize=9, fontweight='bold', color='white', family='monospace')
                ax.text(x + 1.8, y, f'{name}:', va='center',
                       fontsize=9, fontweight='bold', color='white')
                ax.text(x + 2.5, y, desc, va='center',
                       fontsize=8, color='white')

            # Usage example
            note_y = 2
            note_rect = FancyBboxPatch((1, note_y - 1), 12, 1.8,
                                      boxstyle="round,pad=0.1",
                                      facecolor='#F0F0F0', edgecolor='#333333', linewidth=1.5)
            ax.add_patch(note_rect)
            ax.text(7, note_y + 0.5, 'CHROUT Example (print "A" to screen):', ha='center', fontsize=10, fontweight='bold')
            ax.text(7, note_y + 0.1, 'LDA #65   ; ASCII code for "A"',
                   ha='center', fontsize=9, family='monospace')
            ax.text(7, note_y - 0.3, 'JSR $FFD2 ; Call CHROUT',
                   ha='center', fontsize=9, family='monospace')
            ax.text(7, note_y - 0.7, 'BASIC equivalent: PRINT CHR$(65)',
                   ha='center', fontsize=9, style='italic')

            filename = 'kernal_jumptable.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'KERNAL Jump Table',
                'description': 'Common KERNAL ROM routines for I/O and file operations'
            })

        # User Port Pinout
        if 'USER PORT' in title_upper or 'USERPORT' in title_upper:
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.set_xlim(0, 12)
            ax.set_ylim(0, 14)
            ax.axis('off')

            ax.text(6, 13, 'C64 User Port Pinout',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(6, 12.5, '24-pin Edge Connector (CIA2 + Serial Bus)',
                   ha='center', fontsize=11, style='italic', color='#666666')

            # User port pins
            y_start = 11
            pins = [
                ('Pin 1', 'GND', 'Ground', '#666666'),
                ('Pin 2', '+5V', 'Power (+5 volts)', '#E63946'),
                ('Pin 3', '/RESET', 'Reset (active low)', '#F4A261'),
                ('Pin 4-11', 'PB0-PB7', 'CIA2 Port B (Data)', '#4A90E2'),
                ('Pin 12', 'GND', 'Ground', '#666666'),
                ('Pin C', 'PB0', 'CIA2 Port B bit 0', '#4A90E2'),
                ('Pin D', 'PB1', 'CIA2 Port B bit 1', '#4A90E2'),
                ('Pin E', 'PB2', 'CIA2 Port B bit 2', '#4A90E2'),
                ('Pin F', 'PB3', 'CIA2 Port B bit 3', '#4A90E2'),
                ('Pin H', 'PB4', 'CIA2 Port B bit 4', '#4A90E2'),
                ('Pin J', 'PB5', 'CIA2 Port B bit 5', '#4A90E2'),
                ('Pin K', 'PB6', 'CIA2 Port B bit 6', '#4A90E2'),
                ('Pin L', 'PB7', 'CIA2 Port B bit 7', '#4A90E2'),
                ('Pin M', 'PA2', 'CIA2 Port A bit 2', '#50C878'),
                ('Pin N', 'GND', 'Ground', '#666666'),
            ]

            # Show first 8 pins
            for i in range(min(8, len(pins))):
                pin, signal, desc, color = pins[i]
                y = y_start - (i * 0.7)
                rect = FancyBboxPatch((1, y - 0.3), 10, 0.55,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(6, y, f'{pin}: {signal} - {desc}', ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')

            # CIA2 registers
            y_reg = 4.5
            ax.text(6, y_reg + 0.5, 'CIA2 User Port Registers:', ha='center', fontsize=11, fontweight='bold')

            reg_info = [
                '$DD00 (56576): Port A data register',
                '$DD01 (56577): Port B data register (8 data pins)',
                '$DD02 (56578): Port A direction (0=input, 1=output)',
                '$DD03 (56579): Port B direction (0=input, 1=output)',
            ]

            for i, info in enumerate(reg_info):
                y = y_reg - (i * 0.35)
                ax.text(6, y, info, ha='center', fontsize=9, family='monospace')

            # Example
            note_y = 1.8
            note_rect = FancyBboxPatch((1, note_y - 0.8), 10, 1.5,
                                      boxstyle="round,pad=0.1",
                                      facecolor='#F0F0F0', edgecolor='#333333', linewidth=1.5)
            ax.add_patch(note_rect)
            ax.text(6, note_y + 0.3, 'Output Example (set all Port B pins high):', ha='center', fontsize=10, fontweight='bold')
            ax.text(6, note_y - 0.1, 'LDA #$FF : STA $DD03  ; Set Port B as output',
                   ha='center', fontsize=9, family='monospace')
            ax.text(6, note_y - 0.4, 'LDA #$FF : STA $DD01  ; Set all pins high',
                   ha='center', fontsize=9, family='monospace')

            filename = 'user_port.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'User Port Pinout',
                'description': '24-pin edge connector with CIA2 port access for expansion'
            })

        # Datasette Tape Format
        if 'DATASETTE' in title_upper or 'TAPE' in title_upper or 'CASSETTE' in title_upper:
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.set_xlim(0, 12)
            ax.set_ylim(0, 14)
            ax.axis('off')

            ax.text(6, 13, 'C64 Datasette Tape Format',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(6, 12.5, 'Cassette Data Storage',
                   ha='center', fontsize=11, style='italic', color='#666666')

            # Tape structure
            y_start = 11
            tape_sections = [
                ('Leader Tone', '2 seconds', 'Constant signal for sync', '#4A90E2'),
                ('Sync Pattern', '~9000 pulses', 'Synchronization bytes', '#50C878'),
                ('Data Block', 'Variable', 'Program/data bytes', '#E76F51'),
                ('Checksum', '1 byte', 'Data verification', '#F4A261'),
                ('Trailer', 'Short pause', 'End of block marker', '#9D4EDD'),
            ]

            for i, (name, duration, desc, color) in enumerate(tape_sections):
                y = y_start - (i * 0.9)
                rect = FancyBboxPatch((1, y - 0.35), 10, 0.7,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(6, y + 0.1, name, ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white')
                ax.text(6, y - 0.15, f'{duration} - {desc}', ha='center', va='center',
                       fontsize=8, color='white')

            # Data encoding
            y_encode = 6.5
            ax.text(6, y_encode + 0.5, 'Pulse Encoding:', ha='center', fontsize=11, fontweight='bold')

            encode_info = [
                'Short pulse (~296 µs) = 0 bit',
                'Medium pulse (~440 µs) = 1 bit',
                'Long pulse (~672 µs) = End marker',
                'Tape speed: 300 baud (300 bits/second)',
            ]

            for i, info in enumerate(encode_info):
                y = y_encode - (i * 0.35)
                ax.text(6, y, f'• {info}', ha='center', fontsize=9)

            # BASIC commands
            note_y = 3.5
            note_rect = FancyBboxPatch((1, note_y - 1.2), 10, 2.2,
                                      boxstyle="round,pad=0.1",
                                      facecolor='#F0F0F0', edgecolor='#333333', linewidth=1.5)
            ax.add_patch(note_rect)
            ax.text(6, note_y + 0.7, 'BASIC Tape Commands:', ha='center', fontsize=10, fontweight='bold')
            ax.text(6, note_y + 0.3, 'LOAD - Load program from tape',
                   ha='center', fontsize=9, family='monospace')
            ax.text(6, note_y - 0.1, 'SAVE "filename" - Save program to tape',
                   ha='center', fontsize=9, family='monospace')
            ax.text(6, note_y - 0.5, 'VERIFY - Compare tape to memory',
                   ha='center', fontsize=9, family='monospace')
            ax.text(6, note_y - 0.9, 'Control register: $01 (bit 5 = motor, bits 3-4 = read/write)',
                   ha='center', fontsize=8, style='italic')

            filename = 'datasette_format.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'Datasette Tape Format',
                'description': 'Cassette tape data structure and pulse encoding'
            })

        # Zero Page Memory Map
        if 'ZERO' in title_upper and 'PAGE' in title_upper:
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 12)
            ax.axis('off')

            ax.text(7, 11.5, 'C64 Zero Page Memory Map ($00-$FF)',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(7, 11, 'Critical System Variables in First 256 Bytes',
                   ha='center', fontsize=11, style='italic', color='#666666')

            # Key zero page locations
            zp_areas = [
                ('$00-$01', 'Processor Port', '#E63946', 10.2),
                ('$02-$0A', 'BASIC Pointers', '#4A90E2', 9.4),
                ('$14-$24', 'Kernal Variables', '#50C878', 8.6),
                ('$2B-$2C', 'BASIC Start', '#F4A261', 7.8),
                ('$2D-$2E', 'BASIC End', '#9D4EDD', 7.0),
                ('$37-$38', 'Array Start', '#2A9D8F', 6.2),
                ('$91', 'Stop Key Flag', '#E76F51', 5.4),
                ('$C0-$C5', 'Float Acc', '#4ECDC4', 4.6),
                ('$D3', 'Cursor Column', '#F4A261', 3.8),
                ('$D6', 'Cursor Row', '#50C878', 3.0),
            ]

            for addr, label, color, y in zp_areas:
                rect = FancyBboxPatch((1, y-0.3), 12, 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(7, y, f'{addr}: {label}', ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white')

            # Example code
            ax.text(7, 1.8, 'Assembly Example:', ha='center', fontsize=10, fontweight='bold')
            ax.text(7, 1.4, 'LDA $D020  ; Read border color', ha='center', fontsize=9, family='monospace')
            ax.text(7, 1.0, 'STA $FB    ; Store in zero page', ha='center', fontsize=9, family='monospace')
            ax.text(7, 0.6, 'LDA ($FB),Y ; Indirect indexed', ha='center', fontsize=9, family='monospace')

            filename = 'zero_page.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'Zero Page Memory Map',
                'description': 'Critical system variables in $00-$FF with addressing modes'
            })

        # Stack Visualization
        if 'STACK' in title_upper:
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 12)
            ax.axis('off')

            ax.text(7, 11.5, '6502 Stack ($0100-$01FF)',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(7, 11, 'Stack Pointer (SP) grows downward from $01FF',
                   ha='center', fontsize=11, style='italic', color='#666666')

            # Stack visualization
            stack_y = 9.5
            for i in range(8):
                addr = f'$01{255-i:02X}'
                y = stack_y - (i * 0.6)

                if i == 0:
                    color = '#E63946'
                    label = f'{addr} ← SP (Stack Pointer)'
                elif i < 3:
                    color = '#F4A261'
                    label = f'{addr} (Return Address)'
                else:
                    color = '#CBD5E0'
                    label = f'{addr} (Free)'

                rect = FancyBboxPatch((2, y-0.25), 10, 0.4,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(7, y, label, ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white' if i < 3 else '#2D3748')

            # Operations
            ax.text(7, 4.5, 'Stack Operations:', ha='center', fontsize=11, fontweight='bold')
            ops = [
                'PHA - Push Accumulator (SP--)',
                'PLA - Pull Accumulator (SP++)',
                'PHP - Push Status Register',
                'PLP - Pull Status Register',
                'JSR - Push Return Address (2 bytes)',
                'RTS - Pull Return Address',
            ]
            for i, op in enumerate(ops):
                ax.text(7, 3.9 - (i * 0.4), op, ha='center', fontsize=9, family='monospace')

            filename = 'stack_diagram.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'Stack Visualization',
                'description': '256-byte stack with push/pull operations and SP management'
            })

        # PETSCII Character Set
        if 'PETSCII' in title_upper:
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 12)
            ax.axis('off')

            ax.text(7, 11.5, 'PETSCII Character Set',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(7, 11, 'Commodore 64 Character Encoding (256 Characters)',
                   ha='center', fontsize=11, style='italic', color='#666666')

            # Character ranges
            ranges = [
                ('$00-$1F', 'Control Characters', '#E63946', 10.0),
                ('$20-$3F', 'Uppercase + Symbols', '#4A90E2', 9.3),
                ('$40-$5F', 'Lowercase + Graphics', '#50C878', 8.6),
                ('$60-$7F', 'Uppercase + Graphics', '#F4A261', 7.9),
                ('$80-$9F', 'Control (Reverse)', '#E76F51', 7.2),
                ('$A0-$BF', 'Graphics + Symbols', '#9D4EDD', 6.5),
                ('$C0-$DF', 'Uppercase (Reverse)', '#2A9D8F', 5.8),
                ('$E0-$FF', 'Lowercase (Reverse)', '#4ECDC4', 5.1),
            ]

            for addr, label, color, y in ranges:
                rect = FancyBboxPatch((2, y-0.3), 10, 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(7, y, f'{addr}: {label}', ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white')

            # Common characters
            ax.text(7, 3.8, 'Common Characters:', ha='center', fontsize=10, fontweight='bold')
            chars = [
                '$20 = Space   $41 = A   $61 = a',
                '$30-$39 = Digits 0-9',
                '$13 = HOME   $14 = DEL   $1D = Cursor Right',
                '$91 = Cursor Up   $11 = Cursor Down',
            ]
            for i, char in enumerate(chars):
                ax.text(7, 3.2 - (i * 0.4), char, ha='center', fontsize=9, family='monospace')

            # Screen codes note
            ax.text(7, 1.2, 'Note: Screen codes differ from PETSCII codes!',
                   ha='center', fontsize=9, style='italic', color='#E63946', fontweight='bold')
            ax.text(7, 0.7, 'Use POKE for screen memory, CHR$ for PRINT',
                   ha='center', fontsize=8, style='italic', color='#666666')

            filename = 'petscii_chart.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'PETSCII Character Set',
                'description': 'Complete 256-character encoding with control codes and graphics'
            })

        # Joystick Port Wiring
        if 'JOYSTICK' in title_upper:
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 12)
            ax.axis('off')

            ax.text(7, 11.5, 'C64 Joystick Port Wiring',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(7, 11, '9-Pin D-Sub Connector (Atari Standard)',
                   ha='center', fontsize=11, style='italic', color='#666666')

            # Pin layout
            pins = [
                ('Pin 1', 'Up', '#4A90E2', 2, 9.5),
                ('Pin 2', 'Down', '#4A90E2', 5, 9.5),
                ('Pin 3', 'Left', '#50C878', 2, 8.5),
                ('Pin 4', 'Right', '#50C878', 5, 8.5),
                ('Pin 5', 'POT Y', '#9D4EDD', 8.5, 9.5),
                ('Pin 6', 'Fire', '#E63946', 2, 7.5),
                ('Pin 7', '+5V', '#F4A261', 5, 7.5),
                ('Pin 8', 'Ground', '#2D3748', 8.5, 8.5),
                ('Pin 9', 'POT X', '#9D4EDD', 8.5, 7.5),
            ]

            for pin, label, color, x, y in pins:
                rect = FancyBboxPatch((x, y-0.3), 2.5, 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                text_color = 'white' if color != '#F4A261' else 'black'
                ax.text(x+1.25, y, f'{pin}: {label}', ha='center', va='center',
                       fontsize=9, fontweight='bold', color=text_color)

            # CIA registers
            ax.text(7, 6.0, 'CIA Registers:', ha='center', fontsize=11, fontweight='bold')
            ax.text(7, 5.5, 'Port 1: $DC00 (CIA1 Data Port A)', ha='center', fontsize=9, family='monospace')
            ax.text(7, 5.1, 'Port 2: $DC01 (CIA1 Data Port B)', ha='center', fontsize=9, family='monospace')

            # Read example
            ax.text(7, 4.3, 'Assembly Example (Read Port 2):', ha='center', fontsize=10, fontweight='bold')
            code = [
                'LDA $DC00  ; Read joystick port 2',
                'AND #$1F   ; Mask direction + fire bits',
                'CMP #$1F   ; Check if centered (all high)',
                'BEQ NoMove ; No joystick movement',
            ]
            for i, line in enumerate(code):
                ax.text(7, 3.7 - (i * 0.35), line, ha='center', fontsize=8, family='monospace')

            # Bit mapping
            ax.text(7, 1.8, 'Bit Mapping: 0=Active, 1=Inactive', ha='center', fontsize=9, fontweight='bold')
            bits = 'Bit 0=Up  1=Down  2=Left  3=Right  4=Fire'
            ax.text(7, 1.4, bits, ha='center', fontsize=8, family='monospace')

            filename = 'joystick_wiring.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'Joystick Port Wiring',
                'description': '9-pin connector layout with CIA register mapping and read example'
            })

        # Color Palette Chart
        if 'COLOR' in title_upper or 'COLOUR' in title_upper:
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 12)
            ax.axis('off')

            ax.text(7, 11.5, 'C64 Color Palette (16 Colors)',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(7, 11, 'VIC-II Color Values ($D020-$D021 Border/Background)',
                   ha='center', fontsize=11, style='italic', color='#666666')

            # C64 color palette (approximate RGB)
            c64_colors = [
                (0, 'Black', '#000000'),
                (1, 'White', '#FFFFFF'),
                (2, 'Red', '#880000'),
                (3, 'Cyan', '#AAFFEE'),
                (4, 'Purple', '#CC44CC'),
                (5, 'Green', '#00CC55'),
                (6, 'Blue', '#0000AA'),
                (7, 'Yellow', '#EEEE77'),
                (8, 'Orange', '#DD8855'),
                (9, 'Brown', '#664400'),
                (10, 'Lt Red', '#FF7777'),
                (11, 'Dk Grey', '#333333'),
                (12, 'Grey', '#777777'),
                (13, 'Lt Green', '#AAFF66'),
                (14, 'Lt Blue', '#0088FF'),
                (15, 'Lt Grey', '#BBBBBB'),
            ]

            # Display colors in 4x4 grid
            for i, (num, name, rgb) in enumerate(c64_colors):
                row = i // 4
                col = i % 4
                x = 1.5 + (col * 3)
                y = 9.5 - (row * 1.8)

                # Color box
                rect = FancyBboxPatch((x, y-0.5), 2.5, 0.8,
                                     boxstyle="round,pad=0.05",
                                     facecolor=rgb, edgecolor='black', linewidth=2)
                ax.add_patch(rect)

                # Text (choose contrasting color)
                text_color = 'white' if num in [0, 2, 6, 9, 11] else 'black'
                ax.text(x+1.25, y, f'{num}: {name}', ha='center', va='center',
                       fontsize=9, fontweight='bold', color=text_color)

            # Usage examples
            ax.text(7, 2.5, 'BASIC Examples:', ha='center', fontsize=10, fontweight='bold')
            examples = [
                'POKE 53280,0  : REM Border = Black',
                'POKE 53281,6  : REM Background = Blue',
                'PRINT CHR$(18); : REM Reverse On (color 18)',
            ]
            for i, ex in enumerate(examples):
                ax.text(7, 2.0 - (i * 0.35), ex, ha='center', fontsize=8, family='monospace')

            ax.text(7, 0.5, 'Note: Colors appear differently on real hardware vs emulators',
                   ha='center', fontsize=8, style='italic', color='#666666')

            filename = 'color_palette.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'C64 Color Palette',
                'description': 'All 16 colors with POKE values and BASIC examples'
            })

        # Waveform Types
        if 'WAVEFORM' in title_upper or ('SOUND' in title_upper and 'WAVE' in title_upper):
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 12)
            ax.axis('off')

            ax.text(7, 11.5, 'SID Waveform Types',
                   ha='center', fontsize=16, fontweight='bold')
            ax.text(7, 11, 'Four Basic Waveforms ($D404, $D40B, $D412)',
                   ha='center', fontsize=11, style='italic', color='#666666')

            import numpy as np

            # Waveform data
            waveforms = [
                ('Triangle', '#4A90E2', 9.5, lambda x: np.abs(2 * (x % 1) - 1) * 2 - 1),
                ('Sawtooth', '#50C878', 7.5, lambda x: 2 * (x % 1) - 1),
                ('Pulse', '#E76F51', 5.5, lambda x: np.where((x % 1) < 0.5, 1, -1)),
                ('Noise', '#9D4EDD', 3.5, lambda x: np.random.uniform(-1, 1, len(x))),
            ]

            for name, color, y_center, wave_func in waveforms:
                # Generate waveform
                x_vals = np.linspace(0, 3, 300)
                y_vals = wave_func(x_vals) * 0.6 + y_center

                # Plot waveform
                ax.plot(1.5 + x_vals * 3, y_vals, color=color, linewidth=2.5)

                # Background box
                rect = FancyBboxPatch((1.3, y_center-0.8), 10, 1.4,
                                     boxstyle="round,pad=0.05",
                                     facecolor='#F7FAFC', edgecolor=color, linewidth=2)
                ax.add_patch(rect)

                # Label
                ax.text(12.2, y_center, name, fontsize=11, fontweight='bold', color=color, va='center')

            # Control bits
            ax.text(7, 1.5, 'Control Register Bits:', ha='center', fontsize=10, fontweight='bold')
            bits = [
                'Bit 4: Triangle  Bit 5: Sawtooth  Bit 6: Pulse  Bit 7: Noise',
                'Example: LDA #$11 / STA $D404  ; Triangle wave + Gate',
            ]
            for i, bit in enumerate(bits):
                ax.text(7, 1.0 - (i * 0.35), bit, ha='center', fontsize=8, family='monospace')

            filename = 'waveforms.png'
            filepath = images_dir / filename
            plt.tight_layout()
            plt.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            diagrams.append({
                'filename': filename,
                'path': f"../assets/images/articles/{filename}",
                'title': 'SID Waveform Types',
                'description': 'Four basic waveforms with control register bits'
            })

        return diagrams
